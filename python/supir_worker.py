"""
SUPIR Worker - JSON IPC subprocess for image upscaling.

Communication protocol:
  stdin  <- JSON commands (one per line)
  stdout -> JSON responses (one per line)
  stderr -> Library output, tqdm progress, logs

Commands:
  {"command": "ping"}
  {"command": "shutdown"}
  {"command": "process", "input": "...", "output": "...", "device": "cuda"|"dml"|"cpu", ...}
"""

import json
import sys
import os
import traceback
import logging
from pathlib import Path
import faulthandler

# Reconfigure encoding to UTF-8 BEFORE capturing stdout reference.
# On Japanese Windows, the default encoding is cp932 (Shift-JIS).
# If an early error sends JSON via _original_stdout before main() runs,
# non-ASCII characters would be encoded in cp932 instead of UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

# Redirect stdout to stderr so library prints don't corrupt JSON IPC
_original_stdout = sys.stdout
sys.stdout = sys.stderr

# Dump Python traceback on fatal native crashes (access violation, etc.)
faulthandler.enable()


def send(payload: dict) -> None:
    """Send a JSON response on the original stdout."""
    _original_stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _original_stdout.flush()


class _XformersBlocker:
    """Import hook that blocks xformers on non-CUDA devices.

    Prevents DLL load errors from CUDA-specific native extensions
    when running on CPU or DirectML devices.
    xformers is CUDA-only and its C extension (_C) fails to load
    without CUDA runtime, causing cascading import errors in diffusers.

    Uses the modern find_spec API (Python 3.4+) instead of the
    deprecated find_module/load_module which are removed in Python 3.12+.
    """

    def __init__(self, device: str):
        """Initialize the blocker with the current device type."""
        self._device = device

    def find_module(self, fullname, path=None):
        """Legacy fallback for Python < 3.4 compatibility."""
        if fullname == "xformers" or fullname.startswith("xformers."):
            return self
        return None

    def load_module(self, fullname):
        """Legacy fallback: raise ImportError for blocked modules."""
        raise ImportError(
            f"xformers is blocked on {self._device} device (CUDA-only extension)"
        )

    def find_spec(self, fullname, path=None, target=None):
        """Modern API: intercept xformers module lookups (Python 3.4+)."""
        if fullname == "xformers" or fullname.startswith("xformers."):
            raise ImportError(
                f"xformers is blocked on {self._device} device (CUDA-only extension)"
            )
        return None


def _get_available_memory_bytes() -> int:
    """Return approximate available physical memory in bytes.

    Uses psutil if available, otherwise falls back to ctypes on Windows.
    Returns 0 if detection fails (caller should treat as 'unknown').
    """
    # Try psutil first (most reliable cross-platform).
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass

    # Fallback: Win32 GlobalMemoryStatusEx via ctypes.
    try:
        import ctypes
        import ctypes.wintypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        mem = _MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            return mem.ullAvailPhys
    except Exception:
        pass

    return 0


_SAFETENSORS_DTYPE_MAP = None  # lazily initialised in _patch_safetensors_load


def _load_safetensors_streaming(filename, device="cpu", logger=None):
    """Load a safetensors file one tensor at a time via seek+read.

    This avoids both:
      - mmap (which crashes in torch+cpu on Windows)
      - loading the entire file into memory (which causes MemoryError
        for ~10 GB files on low-RAM machines)

    The safetensors binary format is:
      [8 bytes header_size (LE uint64)]
      [header_size bytes JSON header]
      [tensor data …]

    Each tensor entry in the JSON header has:
      {"dtype": "F32", "shape": [d0, d1, …], "data_offsets": [start, end]}
    where offsets are relative to the first byte after the header.

    Sends periodic heartbeat progress messages via stdout so that the
    GUI side can reset its timeout during long-running loads on low-memory
    systems where page-file I/O makes reading very slow.
    """
    import struct
    import json as _json
    import time as _time
    import torch

    global _SAFETENSORS_DTYPE_MAP
    if _SAFETENSORS_DTYPE_MAP is None:
        _SAFETENSORS_DTYPE_MAP = {
            "BOOL": torch.bool,
            "U8": torch.uint8,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "F64": torch.float64,
        }

    basename = os.path.basename(filename)
    file_size = os.path.getsize(filename)
    file_mb = file_size / (1024 * 1024)

    with open(filename, "rb") as f:
        # --- header ---
        raw_header_size = f.read(8)
        if len(raw_header_size) < 8:
            raise ValueError(f"Invalid safetensors file (too short): {filename}")
        header_size = struct.unpack("<Q", raw_header_size)[0]
        # Validate header_size before allocating (B1-BUG-02: corrupt files)
        _MAX_HEADER = 100 * 1024 * 1024  # 100 MB — no legitimate header is larger
        if header_size > _MAX_HEADER or 8 + header_size > file_size:
            raise ValueError(
                f"Invalid safetensors header_size {header_size} "
                f"(file_size={file_size}): {filename}"
            )
        header_bytes = f.read(header_size)
        if len(header_bytes) < header_size:
            raise ValueError(f"Truncated safetensors header: {filename}")
        header = _json.loads(header_bytes)

        data_base_offset = 8 + header_size  # absolute offset of tensor data

        # Sort tensors by disk offset for sequential I/O (A1-#2)
        tensor_entries = dict(
            sorted(
                ((k, v) for k, v in header.items() if k != "__metadata__"),
                key=lambda kv: kv[1]["data_offsets"][0],
            )
        )
        total_tensors = len(tensor_entries)

        # Heartbeat: send progress every _HEARTBEAT_INTERVAL seconds so the
        # GUI can reset its read-timeout while streaming is still working.
        _HEARTBEAT_INTERVAL = 30  # seconds
        last_heartbeat = _time.monotonic()

        result = {}
        for idx, (name, meta) in enumerate(tensor_entries.items()):
            dtype_str = meta["dtype"]
            shape = meta["shape"]

            # Validate data_offsets (B1-BUG-03)
            offsets = meta.get("data_offsets")
            if not (isinstance(offsets, (list, tuple)) and len(offsets) == 2):
                raise ValueError(f"Missing or malformed data_offsets for tensor '{name}'")
            start, end = int(offsets[0]), int(offsets[1])
            if end < start:
                raise ValueError(f"Negative byte_len ({end - start}) for tensor '{name}'")
            byte_len = end - start
            if data_base_offset + end > file_size:
                raise ValueError(
                    f"Tensor '{name}' offsets [{start},{end}] exceed file size {file_size}"
                )

            torch_dtype = _SAFETENSORS_DTYPE_MAP.get(dtype_str)
            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported safetensors dtype '{dtype_str}' "
                    f"for tensor '{name}'"
                )

            f.seek(data_base_offset + start)
            if byte_len == 0:
                tensor = torch.empty(shape, dtype=torch_dtype)
            else:
                # Use readinto to avoid an extra bytes→bytearray copy (A1-#1)
                buf = bytearray(byte_len)
                n_read = f.readinto(buf)
                if n_read < byte_len:
                    raise ValueError(
                        f"Truncated tensor data for '{name}' in {filename}"
                    )
                # .clone() detaches from the bytearray buffer so that:
                #   1) the raw bytes can be garbage-collected immediately
                #   2) the tensor has a standard UntypedStorage that is
                #      compatible with torch-directml's .to() method
                tensor = torch.frombuffer(
                    buf, dtype=torch_dtype
                ).reshape(shape).clone()
                del buf  # release buffer early

            result[name] = tensor

            # Periodic heartbeat to prevent GUI timeout.
            now = _time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                pct = int((idx + 1) / total_tensors * 100)
                send({
                    "status": "progress",
                    "message": (
                        f"ストリーミング読み込み中: {basename} "
                        f"({pct}%, {idx + 1}/{total_tensors} tensors)"
                    ),
                    "phase": "streaming_load",
                })
                last_heartbeat = now

    if device != "cpu":
        result = {k: v.to(device) for k, v in result.items()}

    return result


def _patch_safetensors_load(device: str, logger):
    """Patch safetensors.torch.load_file based on device and system memory.

    On Windows with torch 2.4.x+cpu / DirectML, the default mmap-based
    loading in safetensors can trigger an ACCESS_VIOLATION inside
    torch.UntypedStorage.__getitem__ when reading large safetensors files
    (e.g. the ~10 GB CLIP ViT-bigG checkpoint).

    Strategy:
      - CUDA     : No patch needed; mmap works correctly with CUDA torch.
      - Non-CUDA :
          * Enough memory (>= file × 2.5)  → full byte-read (fast)
          * Insufficient memory             → streaming tensor-by-tensor
            read via seek+read (low peak memory, no mmap)
    """
    if device == "cuda":
        logger.info("CUDA device: safetensors mmap loading (default)")
        return

    try:
        import safetensors.torch as st_torch
        from safetensors import deserialize as _deserialize

        _original_load_file = st_torch.load_file
        _view2torch = getattr(st_torch, "_view2torch", None)

        # Memory multiplier: we need room for the raw bytes *plus* the
        # deserialized tensors plus Python object overhead.
        _MEM_HEADROOM_FACTOR = 3.0

        def _load_via_bytes(filename, device="cpu"):
            """Full byte-read → deserialize.  Fast but needs ~2× file RAM."""
            with open(filename, "rb") as f:
                data = f.read()
            flat = _deserialize(data)
            del data  # free raw bytes before any fallback (B1-BUG-08)
            if _view2torch is not None:
                result = _view2torch(flat)
            else:
                # Fall back to streaming loader instead of mmap-based original
                return _load_safetensors_streaming(filename, device=device, logger=logger)
            if device != "cpu":
                import torch
                result = {k: v.to(device) for k, v in result.items()}
            return result

        def _smart_load_file(filename, device="cpu"):
            """Smart loader: pick strategy based on file size vs. memory."""
            file_size = os.path.getsize(filename)
            avail = _get_available_memory_bytes()
            file_mb = file_size / (1024 * 1024)

            if avail > 0 and avail >= file_size * _MEM_HEADROOM_FACTOR:
                logger.info(
                    f"safetensors: byte-read {file_mb:.0f} MB "
                    f"(avail {avail / (1024**3):.1f} GB)"
                )
                return _load_via_bytes(filename, device)
            else:
                logger.info(
                    f"safetensors: streaming-read {file_mb:.0f} MB "
                    f"(avail {avail / (1024**3):.1f} GB)"
                )
                return _load_safetensors_streaming(
                    filename, device, logger=logger
                )

        st_torch.load_file = _smart_load_file
        logger.info("Non-CUDA device: safetensors mmap bypassed (smart loader)")
    except Exception as e:
        # If anything goes wrong during patching, silently skip so that
        # the original behaviour is preserved.
        logger.warning(f"safetensors patch failed, using default loader: {e}")


_directml_storage_patched = False


def _patch_directml_storage_copy(logger):
    """Monkey-patch torch-directml's ``_StorageBase.privateuseone`` so that
    ``UntypedStorage.copy_()`` failures (``UnicodeDecodeError``) are handled
    transparently.

    The original ``_dml`` method does::

        untyped_storage = torch.UntypedStorage(self.size())
        untyped_storage.copy_(self, non_blocking)

    On some storage layouts the native ``copy_()`` fails with a
    ``UnicodeDecodeError``.  The patched version catches this and falls
    back to writing the bytes via ``ctypes.memmove``.

    Must be called AFTER ``import torch_directml`` so that
    ``_StorageBase.privateuseone`` has been registered.
    """
    global _directml_storage_patched
    if _directml_storage_patched:
        return
    try:
        import torch
        from torch.storage import _StorageBase

        _original_dml = getattr(_StorageBase, "privateuseone", None)
        if _original_dml is None:
            logger.info("_dml patch: no privateuseone method found, skipping")
            return

        import ctypes

        def _safe_dml(self, dev=None, non_blocking=False, **kwargs):
            """Safe replacement for torch-directml's _dml.

            Tries the original _dml first; on UnicodeDecodeError rebuilds
            the source storage from raw bytes via ctypes and retries.
            If even that fails, manually allocates a DML storage and
            copies bytes into it via a fresh CPU intermediary.
            """
            # --- Try 1: original path ---
            try:
                return _original_dml(self, dev=dev, non_blocking=non_blocking, **kwargs)
            except UnicodeDecodeError:
                pass

            nbytes = self.size()
            if nbytes == 0:
                # Zero-byte storage: allocate empty on DML
                try:
                    return _original_dml(
                        torch.UntypedStorage(0),
                        dev=dev, non_blocking=non_blocking, **kwargs,
                    )
                except UnicodeDecodeError:
                    return torch.UntypedStorage(0)

            # --- Try 2: rebuild source from raw bytes, then original path ---
            src_ptr = self.data_ptr()
            buf = (ctypes.c_char * nbytes)()
            ctypes.memmove(buf, src_ptr, nbytes)
            clean = torch.UntypedStorage(nbytes)
            ctypes.memmove(clean.data_ptr(), buf, nbytes)
            try:
                return _original_dml(
                    clean, dev=dev, non_blocking=non_blocking, **kwargs,
                )
            except UnicodeDecodeError:
                pass

            # --- Try 3: manual DML allocation + copy_() from clean ---
            # Replicate what _original_dml does internally, but with
            # the clean (freshly allocated) storage as source.
            try:
                import torch_directml as _tdml
                if dev is None:
                    dml_dev = _tdml.device(_tdml.default_device())
                else:
                    dml_dev = _tdml.device(dev)
                with dml_dev:
                    dml_storage = torch.UntypedStorage(nbytes)
                    dml_storage.copy_(clean, non_blocking)
                    return dml_storage
            except UnicodeDecodeError:
                # Even clean → DML copy_() failed; try from_file approach
                import tempfile, os
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
                        tmp_path = f.name
                        f.write(bytes(buf))
                    fresh = torch.UntypedStorage.from_file(
                        tmp_path, shared=False, nbytes=nbytes
                    )
                    return _original_dml(
                        fresh, dev=dev, non_blocking=non_blocking, **kwargs,
                    )
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

        _StorageBase.privateuseone = _safe_dml
        _directml_storage_patched = True
        logger.info("Patched torch-directml _dml with UnicodeDecodeError fallback")
    except Exception as e:
        logger.warning(f"_dml patch failed: {e}")


def _try_tensor_to_dml(tensor, device):
    """Try to move a tensor to DML device.  Returns the tensor on DML if
    successful, or *None* if the transfer failed (UnicodeDecodeError).
    """
    try:
        return tensor.to(device)
    except UnicodeDecodeError:
        return None


def _safe_model_to_device(model, device, logger, send=None):
    """Try to move all parameters and buffers to *device* one-by-one.

    torch-directml has a known bug where its native ``copy_()``
    raises ``UnicodeDecodeError`` after a cumulative amount of data
    has been transferred.  This function transfers tensors individually
    so that the caller can check how many succeeded and decide whether
    to fall back to CPU.
    """
    import torch
    import time as _time

    moved = 0
    errors = 0
    last_heartbeat = _time.monotonic()

    # Count total tensors for progress reporting.
    total = 0
    for module in model.modules():
        for p in module._parameters.values():
            if p is not None:
                total += 1
        for b in module._buffers.values():
            if b is not None:
                total += 1

    for module in model.modules():
        for name in list(module._parameters.keys()):
            p = module._parameters[name]
            if p is None:
                continue
            if hasattr(p, "device") and str(p.device).startswith("privateuseone"):
                moved += 1
                continue
            result = _try_tensor_to_dml(p.data, device)
            if result is not None:
                module._parameters[name] = torch.nn.Parameter(
                    result, requires_grad=p.requires_grad
                )
                moved += 1
            else:
                errors += 1

            # Heartbeat every 30 seconds
            now = _time.monotonic()
            if send is not None and now - last_heartbeat >= 30:
                done = moved + errors
                pct = int(done / total * 100) if total > 0 else 0
                send({
                    "status": "progress",
                    "message": f"モデル転送中: {pct}% ({done}/{total} tensors)",
                    "phase": "model_to_device",
                })
                last_heartbeat = now

        for name in list(module._buffers.keys()):
            b = module._buffers[name]
            if b is None:
                continue
            if hasattr(b, "device") and str(b.device).startswith("privateuseone"):
                moved += 1
                continue
            result = _try_tensor_to_dml(b, device)
            if result is not None:
                module._buffers[name] = result
                moved += 1
            else:
                errors += 1

            now = _time.monotonic()
            if send is not None and now - last_heartbeat >= 30:
                done = moved + errors
                pct = int(done / total * 100) if total > 0 else 0
                send({
                    "status": "progress",
                    "message": f"モデル転送中: {pct}% ({done}/{total} tensors)",
                    "phase": "model_to_device",
                })
                last_heartbeat = now

    # Update lightning_fabric's DeviceDtypeModuleMixin._device
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    for module in model.modules():
        if hasattr(module, "_device"):
            module._device = torch_device

    logger.info(
        f"DML transfer: {moved}/{total} tensors transferred"
        + (f" ({errors} failed)" if errors else "")
    )


def _resolve_torch_device(device: str, logger):
    """Resolve the actual torch device based on the device identifier.

    Args:
        device: Device identifier ("cuda", "dml", or "cpu")
        logger: Logger instance

    Returns:
        Tuple of (actual_device, supir_device_str).
        actual_device: object passed to tensor.to() / model.to()
        supir_device_str: string passed to create_SUPIR_model ("cuda" or "cpu")
    """
    if device == "cuda":
        return "cuda", "cuda"

    if device == "dml":
        # --- SUPIR-GUI patch ---
        # torch-directml has a known bug where the DML runtime enters a
        # corrupted state after transferring ~800 tensors, causing ALL
        # subsequent tensor copy_() operations to fail with
        # UnicodeDecodeError.  Because the SUPIR model has ~3,500 tensors,
        # DML transfer never succeeds.  We therefore skip DML entirely and
        # run inference on CPU.
        logger.info(
            "DML requested but torch-directml has a known tensor transfer "
            "bug (UnicodeDecodeError after ~800 tensors).  Using CPU "
            "inference instead."
        )
        return "cpu", "cpu"

    return "cpu", "cpu"


def _resolve_supir_repo_root() -> str:
    """Resolve the SUPIR repository root directory.

    SUPIR is cloned under the app's lib directory and added to sys.path via ._pth.
    We locate it from the installed SUPIR package path so we can reference the
    repo's options/*.yaml regardless of the current working directory.
    """
    import SUPIR as supir_pkg

    supir_pkg_dir = Path(supir_pkg.__file__).resolve().parent
    return str(supir_pkg_dir.parent)


def _resolve_supir_config_path(repo_root: str, use_tiling: bool) -> str:
    """Pick the config path from env vars, falling back to repo options."""
    config_v0 = os.environ.get("SUPIR_GUI_CONFIG_V0", "")
    config_v0_tiled = os.environ.get("SUPIR_GUI_CONFIG_V0_TILED", "")

    if use_tiling and config_v0_tiled and os.path.isfile(config_v0_tiled):
        return config_v0_tiled
    if (not use_tiling) and config_v0 and os.path.isfile(config_v0):
        return config_v0

    name = "SUPIR_v0_tiled.yaml" if use_tiling else "SUPIR_v0.yaml"
    return os.path.join(repo_root, "options", name)


def main() -> int:
    """Main loop: read JSON commands from stdin, process, respond on stdout."""
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TQDM_MININTERVAL"] = "1"

    # CPU thread tuning: avoid oversubscription (A1-#6)
    _physical_cores = max(1, (os.cpu_count() or 4) // 2)
    _threads = str(min(_physical_cores, 8))
    os.environ.setdefault("OMP_NUM_THREADS", _threads)
    os.environ.setdefault("MKL_NUM_THREADS", _threads)

    # Encoding already reconfigured at module level; no need to repeat here.

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger = logging.getLogger("supir_worker")

    model = None
    current_device = None
    actual_device = None
    xformers_blocked = False
    safetensors_patched = False
    current_use_tiling = None
    current_tiled_size = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            send({"status": "error", "message": f"JSON parse error: {exc}"})
            continue

        command = message.get("command")

        if command == "ping":
            send({"status": "ok", "message": "ready"})
            continue

        if command == "shutdown":
            send({"status": "ok", "message": "shutdown"})
            break

        if command != "process":
            send({"status": "error", "message": f"Unknown command: {command}"})
            continue

        try:
            # Parse processing parameters
            input_path = message["input"]
            output_path = message["output"]
            device = message.get("device", "cuda")
            upscale_factor = int(message.get("upscale_factor", 2))
            edm_steps = int(message.get("edm_steps", 20))
            guidance_scale = float(message.get("guidance_scale", 7.5))
            seed = message.get("seed")
            s_stage1 = float(message.get("s_stage1", -1))
            s_stage2 = float(message.get("s_stage2", 1.0))
            s_churn = float(message.get("s_churn", 5))
            s_noise = float(message.get("s_noise", 1.003))
            tiled_size = int(message.get("tiled_size", 512))
            use_tiling = bool(message.get("use_tiling", True))
            output_format = message.get("output_format", "png").lower()

            # Block xformers on non-CUDA devices to prevent DLL load errors
            if device != "cuda" and not xformers_blocked:
                logger.info(
                    f"Non-CUDA device ({device}): blocking xformers imports"
                )
                sys.meta_path.insert(0, _XformersBlocker(device))
                xformers_blocked = True

            # Patch safetensors to avoid mmap crashes on non-CUDA devices
            # Reset patch if device changed (B1-BUG-05)
            if current_device is not None and current_device != device:
                safetensors_patched = False
            if not safetensors_patched:
                _patch_safetensors_load(device, logger)
                safetensors_patched = True

            # Lazy import and model loading
            if (
                model is None
                or current_device != device
                or current_use_tiling != use_tiling
                or current_tiled_size != tiled_size
            ):
                import torch
                from PIL import Image

                # CPU thread tuning (after torch import) (A1-#6)
                torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
                try:
                    torch.set_num_interop_threads(2)
                except RuntimeError:
                    pass  # already set

                # Resolve torch device
                actual_device, supir_device_str = _resolve_torch_device(
                    device, logger
                )
                logger.info(
                    f"Loading SUPIR model... "
                    f"(device={device}, actual={actual_device}, "
                    f"supir_device={supir_device_str})"
                )

                try:
                    send({
                        "status": "progress",
                        "message": "SUPIRモジュールを読み込み中...",
                        "phase": "import",
                    })
                    from SUPIR.util import create_SUPIR_model

                    repo_root = _resolve_supir_repo_root()
                    config_path = _resolve_supir_config_path(
                        repo_root, use_tiling
                    )
                    logger.info(f"SUPIR config: {config_path}")

                    # Ensure relative paths inside SUPIR behave consistently.
                    os.chdir(repo_root)

                    send({
                        "status": "progress",
                        "message": "モデルの重みを読み込み中...",
                        "phase": "load_weights",
                    })
                    # Load SUPIR (v0-Q).
                    # Note: upstream API uses SUPIR_sign (capitalized).
                    model = create_SUPIR_model(
                        config_path,
                        SUPIR_sign="Q",
                    )
                    # Switch to inference mode: disable dropout, use running
                    # batchnorm stats (A1-#4: was missing, causing non-deterministic output)
                    model.eval()

                    # Optional VAE tiling to reduce memory usage.
                    if use_tiling:
                        send({
                            "status": "progress",
                            "message": "VAEタイリングを初期化中...",
                            "phase": "tile_vae",
                        })
                        model.init_tile_vae(
                            encoder_tile_size=tiled_size,
                            decoder_tile_size=64,
                        )

                    # Move model to the resolved device.
                    # Model is already on CPU from the loading step.
                    # If actual_device is not CPU, move the model there.
                    if str(actual_device) != "cpu":
                        send({
                            "status": "progress",
                            "message": f"モデルを{actual_device}に転送中...",
                            "phase": "model_to_device",
                        })
                        model = model.to(actual_device)

                    logger.info(f"SUPIR model loaded on {actual_device}")
                except ImportError as e:
                    raise RuntimeError(
                        f"SUPIRモジュールの読み込みに失敗しました。"
                        f"SUPIRリポジトリが正しくクローンされているか"
                        f"確認してください: {e}"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"SUPIRモデルの初期化に失敗しました。"
                        f"モデルの重みファイルが正しく配置されているか"
                        f"確認してください: {e}"
                    ) from e

                current_device = device
                current_use_tiling = use_tiling
                current_tiled_size = tiled_size

            import torch
            from PIL import Image

            # Validate paths are absolute (B1-BUG-10)
            if not os.path.isabs(input_path):
                raise ValueError(f"input_path must be absolute, got: {input_path}")
            if not os.path.isabs(output_path):
                raise ValueError(f"output_path must be absolute, got: {output_path}")

            # Load input image
            logger.info(f"Processing: {input_path}")
            input_image = Image.open(input_path).convert("RGB")

            # --- CPU inference: limit input size to avoid OOM / crash ---
            MAX_PROCESS_PIXELS = 4096 * 4096  # ~16 MP (disk swap enabled for tiles)
            iw, ih = input_image.size
            process_pixels = (iw * upscale_factor) * (ih * upscale_factor)
            # Use >= to catch exact boundary (B1-BUG-04)
            if process_pixels >= MAX_PROCESS_PIXELS and str(actual_device) == "cpu":
                ratio = (MAX_PROCESS_PIXELS / process_pixels) ** 0.5
                new_w = max(64, int(iw * ratio) // 64 * 64)
                new_h = max(64, int(ih * ratio) // 64 * 64)
                logger.warning(
                    f"Input image {iw}x{ih} (x{upscale_factor}) is too "
                    f"large for CPU inference "
                    f"({iw * upscale_factor}x{ih * upscale_factor}). "
                    f"Resizing to {new_w}x{new_h} before processing."
                )
                send({
                    "status": "progress",
                    "message": (
                        f"画像が大きすぎるため縮小します: "
                        f"{iw}x{ih} → {new_w}x{new_h}"
                    ),
                    "phase": "resize",
                })
                input_image = input_image.resize(
                    (new_w, new_h), Image.BICUBIC
                )

            # SUPIR preprocessing / inference.
            # Upstream PIL2Tensor returns (tensor[C,H,W], h0, w0).
            from SUPIR.util import PIL2Tensor, Tensor2PIL

            lq, h0, w0 = PIL2Tensor(
                input_image,
                upsacle=upscale_factor,
                min_size=1024,
            )
            lq = lq.unsqueeze(0)[:, :3, :, :]
            lq = lq.to(actual_device)

            captions = [""]
            import random as _random
            seed_value = int(seed) if seed is not None else _random.randint(0, 2**31 - 1)

            # inference_mode is faster than no_grad: disables version tracking (A1-#5)
            with torch.inference_mode():
                samples = model.batchify_sample(
                    lq,
                    captions,
                    num_steps=edm_steps,
                    restoration_scale=s_stage1,
                    s_churn=s_churn,
                    s_noise=s_noise,
                    cfg_scale=guidance_scale,
                    seed=seed_value,
                    num_samples=1,
                    control_scale=s_stage2,
                    color_fix_type="Wavelet",
                    use_linear_CFG=True,
                    use_linear_control_scale=False,
                    cfg_scale_start=3.0,
                    control_scale_start=0.0,
                )

            # Free input tensor early (B1-BUG-06)
            del lq

            # Convert to PIL and save. Skip .cpu() — already on CPU (A1-#7)
            result = Tensor2PIL(samples[0], h0, w0)
            del samples

            # Save output image
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            save_kwargs = {}
            if output_format == "jpg" or output_format == "jpeg":
                save_kwargs["quality"] = 95
                save_kwargs["format"] = "JPEG"
            elif output_format == "webp":
                save_kwargs["quality"] = 95
                save_kwargs["format"] = "WebP"
            elif output_format in ("heic", "heif"):
                try:
                    if not getattr(main, '_heif_registered', False):
                        from pillow_heif import register_heif_opener
                        register_heif_opener()
                        main._heif_registered = True
                except (ImportError, OSError) as e:
                    raise RuntimeError(
                        "HEIC形式で保存するには pillow-heif が必要です。"
                        "初期化をやり直すか、PNG/JPEG/WebP を選択してください。"
                    ) from e

                save_kwargs["quality"] = 95
                save_kwargs["format"] = "HEIF"
            else:
                save_kwargs["format"] = "PNG"

            result.save(output_path, **save_kwargs)

            # Free GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"Done: {output_path}")
            send({"status": "ok", "message": "done", "output": output_path})

        except Exception as exc:
            trace = traceback.format_exc()
            logger.error(f"Processing error: {exc}\n{trace}")
            # Free large tensors to prevent accumulation on repeated failures (B1-BUG-06)
            for _vname in ('lq', 'samples', 'result', 'input_image'):
                if _vname in locals():
                    try:
                        del locals()[_vname]
                    except Exception:
                        pass
            import gc
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            send({"status": "error", "message": str(exc), "traceback": trace})

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
