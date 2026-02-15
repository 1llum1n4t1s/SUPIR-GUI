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
    """

    def __init__(self, device: str):
        """Initialize the blocker with the current device type."""
        self._device = device

    def find_module(self, fullname, path=None):
        """Intercept xformers module lookups."""
        if fullname == "xformers" or fullname.startswith("xformers."):
            return self
        return None

    def load_module(self, fullname):
        """Raise ImportError for blocked xformers modules."""
        raise ImportError(
            f"xformers is blocked on {self._device} device (CUDA-only extension)"
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
        try:
            import torch_directml

            dml_device = torch_directml.device()
            logger.info(f"DirectML device initialized: {dml_device}")
            return dml_device, "cpu"
        except ImportError as e:
            logger.warning(
                f"torch-directml import failed: {e}. Falling back to CPU."
            )
            return "cpu", "cpu"
        except Exception as e:
            logger.warning(
                f"DirectML initialization failed: {e}. Falling back to CPU."
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

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    if hasattr(_original_stdout, "reconfigure"):
        _original_stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger = logging.getLogger("supir_worker")

    model = None
    current_device = None
    actual_device = None
    xformers_blocked = False
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

            # Lazy import and model loading
            if (
                model is None
                or current_device != device
                or current_use_tiling != use_tiling
                or current_tiled_size != tiled_size
            ):
                import torch
                from PIL import Image

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
                    from SUPIR.util import create_SUPIR_model

                    repo_root = _resolve_supir_repo_root()
                    config_path = _resolve_supir_config_path(
                        repo_root, use_tiling
                    )
                    logger.info(f"SUPIR config: {config_path}")

                    # Ensure relative paths inside SUPIR behave consistently.
                    os.chdir(repo_root)

                    # Load SUPIR (v0-Q).
                    # Note: upstream API uses SUPIR_sign (capitalized).
                    model = create_SUPIR_model(
                        config_path,
                        SUPIR_sign="Q",
                    )

                    # Optional VAE tiling to reduce memory usage.
                    if use_tiling:
                        model.init_tile_vae(
                            encoder_tile_size=tiled_size,
                            decoder_tile_size=64,
                        )

                    # Move model to the resolved device (CUDA / DirectML / CPU).
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

            # Load input image
            logger.info(f"Processing: {input_path}")
            input_image = Image.open(input_path).convert("RGB")

            # SUPIR preprocessing / inference.
            # Upstream PIL2Tensor returns (tensor[C,H,W], h0, w0).
            from SUPIR.util import PIL2Tensor, Tensor2PIL

            lq, h0, w0 = PIL2Tensor(
                input_image,
                upsacle=upscale_factor,
                min_size=1024,
            )
            lq = lq.unsqueeze(0).to(actual_device)[:, :3, :, :]

            captions = [""]
            seed_value = int(seed) if seed is not None else -1

            with torch.no_grad():
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

            # Convert to PIL and save.
            result = Tensor2PIL(samples[0].cpu(), h0, w0)

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
            elif output_format == "heic" or output_format == "heif":
                try:
                    from pillow_heif import register_heif_opener

                    register_heif_opener()
                except ImportError as e:
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
            send({"status": "error", "message": str(exc), "traceback": trace})

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
