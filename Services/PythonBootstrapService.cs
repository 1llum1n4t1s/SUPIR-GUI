using System.Diagnostics;
using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using SUPIR_GUI.Models;
using static SUPIR_GUI.Services.Logger;

namespace SUPIR_GUI.Services;

/// <summary>
/// 埋め込みPython環境のセットアップとパッケージ管理を行うサービス
/// </summary>
public partial class PythonBootstrapService
{
    /// <summary>ライブラリ保存先のベースディレクトリ</summary>
    private readonly string _libBaseDir;

    /// <summary>Python保存先のベースディレクトリ</summary>
    private readonly string _basePythonDir;

    /// <summary>埋め込みPythonのディレクトリ</summary>
    private readonly string _pythonEmbedDir;

    /// <summary>アプリケーション設定</summary>
    private readonly AppSettings _settings;

    /// <summary>検出されたGPUデバイス種別 ("cuda", "dml", "cpu")</summary>
    private readonly string _detectedDevice;

    /// <summary>埋め込みPythonの最大バージョン</summary>
    private static readonly Version MaxEmbeddedPythonVersion = new(3, 11, 99);

    /// <summary>openai-clip互換のためのsetuptoolsバージョン指定</summary>
    private const string SetuptoolsCompatSpec = "setuptools<81";

    /// <summary>SUPIR関連のチェックポイントを取得するHugging FaceリポジトリID</summary>
    private const string SupirWeightsRepoId = "camenduru/SUPIR";

    /// <summary>SDXLベースモデルのファイル名</summary>
    private const string SdxlBaseCkptFileName = "sd_xl_base_1.0_0.9vae.safetensors";

    /// <summary>SUPIR v0-Q チェックポイントのファイル名</summary>
    private const string SupirV0QCkptFileName = "SUPIR-v0Q.ckpt";

    /// <summary>SUPIR v0-F チェックポイントのファイル名</summary>
    private const string SupirV0FCkptFileName = "SUPIR-v0F.ckpt";

    /// <summary>torch-directml のパッケージ指定 (互換性確保のため固定)</summary>
    private const string TorchDirectMlSpec = "torch-directml==0.2.5.dev240914";

    /// <summary>torch-directml が要求する torch バージョン</summary>
    private const string TorchDirectMlTorchSpec = "torch==2.4.1";

    /// <summary>torch-directml が要求する torchvision バージョン</summary>
    private const string TorchDirectMlTorchVisionSpec = "torchvision==0.19.1";

    /// <summary>HTTP通信用クライアント</summary>
    private static readonly HttpClient SharedHttpClient = new(
        new SocketsHttpHandler { AutomaticDecompression = DecompressionMethods.All })
    { Timeout = TimeSpan.FromSeconds(30) };

    /// <summary>Pythonの実行ファイルパス</summary>
    private string PythonExePath => Path.Combine(_pythonEmbedDir, "python.exe");

    /// <summary>site-packagesのパス</summary>
    private string SitePackagesPath => Path.Combine(_pythonEmbedDir, "Lib", "site-packages");

    /// <summary>依存パッケージインストール済みマーカーファイルのパス</summary>
    private string DepsMarkerFile => Path.Combine(_pythonEmbedDir, ".supir_deps_installed");

    /// <summary>SUPIRリポジトリのクローン先パス</summary>
    private string SupirRepoPath => Path.Combine(_libBaseDir, "SUPIR");

    /// <summary>GUI用に生成するSUPIR設定ファイルのディレクトリ</summary>
    private string SupirConfigDir => Path.Combine(_libBaseDir, "supir_config");

    /// <summary>GUI用SUPIR v0 設定ファイルパス</summary>
    private string SupirConfigV0Path => Path.Combine(SupirConfigDir, "SUPIR_v0_gui.yaml");

    /// <summary>GUI用SUPIR v0 (tiled) 設定ファイルパス</summary>
    private string SupirConfigV0TiledPath => Path.Combine(SupirConfigDir, "SUPIR_v0_tiled_gui.yaml");

    /// <summary>マーカーファイルのバージョン文字列 (デバイス種別を含む)</summary>
    private string CurrentMarkerVersion => $"installed_v8_{_detectedDevice}";

    /// <summary>Pythonバージョンを抽出する正規表現</summary>
    [GeneratedRegex(@"href=""(?<version>\d+\.\d+\.\d+)/""")]
    private static partial Regex VersionRegex();

    /// <summary>pip Collectingパターンの正規表現</summary>
    [GeneratedRegex(@"^Collecting\s+(\S+)")]
    private static partial Regex PipCollectingRegex();

    /// <summary>pip Downloadingパターンの正規表現</summary>
    [GeneratedRegex(@"^Downloading\s+(\S+)")]
    private static partial Regex PipDownloadingRegex();

    /// <summary>pip Installing collected packagesパターンの正規表現</summary>
    [GeneratedRegex(@"^Installing collected packages:\s+(.+)$")]
    private static partial Regex PipInstallingRegex();

    /// <summary>pip Successfully installedパターンの正規表現</summary>
    [GeneratedRegex(@"^Successfully installed\s+(.+)$")]
    private static partial Regex PipSuccessRegex();

    /// <summary>pip Using cachedパターンの正規表現</summary>
    [GeneratedRegex(@"^Using cached\s+(\S+)")]
    private static partial Regex PipUsingCachedRegex();

    /// <summary>
    /// コンストラクタ
    /// </summary>
    /// <param name="settings">アプリケーション設定</param>
    /// <param name="detectedDevice">検出されたGPUデバイス種別</param>
    public PythonBootstrapService(AppSettings settings, string detectedDevice)
    {
        _settings = settings;
        _detectedDevice = detectedDevice;
        // ライブラリの保存先: %LOCALAPPDATA%\SUPIR-GUI\lib
        _libBaseDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "SUPIR-GUI", "lib");
        _basePythonDir = Path.Combine(_libBaseDir, "python");
        _pythonEmbedDir = Path.Combine(_basePythonDir, "python-embed");
    }

    /// <summary>Pythonホームディレクトリ</summary>
    public string PythonHome => _pythonEmbedDir;

    /// <summary>
    /// 環境の初期化を実行する
    /// </summary>
    /// <param name="onProgress">進捗コールバック (step, totalSteps, message, detail)</param>
    /// <returns>初期化が成功したかどうか</returns>
    public async Task<bool> InitializeAsync(Action<int, int, string, string?>? onProgress = null)
    {
        try
        {
            const int totalSteps = 16;

            // 1: Pythonの検出/ダウンロード
            onProgress?.Invoke(1, totalSteps, "Pythonランタイムを確認中...", null);
            if (!await EnsureEmbeddedPythonAsync())
            {
                Log("埋め込みPythonの準備に失敗しました。", LogLevel.Error);
                return false;
            }

            // 2: .pth設定
            onProgress?.Invoke(2, totalSteps, "Python設定を構成中...", null);
            ConfigurePythonPth();

            // 3: マーカーチェック + バージョンチェック + デバイス変更チェック
            onProgress?.Invoke(3, totalSteps, "インストール状態を確認中...", null);
            var needsInstall = !IsDepsInstalled();
            var deviceChanged = IsDeviceChanged();
            var needsUpdate = false;

            if (!needsInstall)
            {
                onProgress?.Invoke(3, totalSteps, "ライブラリの更新を確認中...", null);
                needsUpdate = await CheckForUpdatesAsync();
            }

            if (!needsInstall && !needsUpdate)
            {
                Log("依存パッケージはインストール済み・最新です。整合性を確認します。", LogLevel.Info);

                // スキップパスでも、後から必要になる依存を検証・修復できるようにする
                Directory.CreateDirectory(SitePackagesPath);
                if (!await IsPipAvailableAsync())
                    await InstallPipAsync();

                onProgress?.Invoke(totalSteps, totalSteps, "セットアップを検証中...", null);
                await EnsureSetuptoolsRuntimeAsync();
                await EnsurePillowHeifAsync();

                // モデル設定/チェックポイントが無い場合のみ作成・取得する
                await EnsureSupirModelAssetsIfMissingAsync();

                if (_detectedDevice == GpuDetectionService.DeviceDirectMl)
                {
                    await EnsureTorchDirectMlRuntimeAsync();
                }

                onProgress?.Invoke(totalSteps, totalSteps, "セットアップ完了", null);
                return true;
            }

            if (needsUpdate && !needsInstall)
            {
                Log("ライブラリの更新が見つかりました。アップグレードを実行します。", LogLevel.Info);
            }

            // 4: site-packages準備
            onProgress?.Invoke(4, totalSteps, "site-packages を準備中...", null);
            Directory.CreateDirectory(SitePackagesPath);

            // 5: pip確認
            onProgress?.Invoke(5, totalSteps, "pip を確認中...", null);
            if (!await IsPipAvailableAsync())
            {
                onProgress?.Invoke(5, totalSteps, "pip をインストール中...", "get-pip.py を実行中");
                await InstallPipAsync();
            }

            // 6: ビルドツール
            onProgress?.Invoke(6, totalSteps, "ビルドツールをインストール中...", null);
            await InstallPackagesAsync(
                [SetuptoolsCompatSpec, "wheel"],
                detail => onProgress?.Invoke(6, totalSteps, "ビルドツールをインストール中...", detail));

            // デバイス変更時、またはフルインストール時はPyTorch関連パッケージをクリーンアップ
            if (deviceChanged || needsInstall)
            {
                var reason = deviceChanged ? "GPUデバイス種別の変更" : "依存関係の再インストール";
                Log($"{reason}を検出しました。PyTorch関連パッケージを再インストールします。", LogLevel.Info);
                onProgress?.Invoke(7, totalSteps, "以前のGPUパッケージをクリーンアップ中...", null);
                CleanupPyTorchPackages();
            }

            // 7: PyTorch (デバイス種別に応じたバリアントをインストール)
            if (_detectedDevice == GpuDetectionService.DeviceCuda)
            {
                onProgress?.Invoke(7, totalSteps, "PyTorch (CUDA) をインストール中 (大きなファイルです)...", null);
                await InstallPackagesWithIndexAsync(
                    ["torch", "torchvision"],
                    "https://download.pytorch.org/whl/cu121",
                    detail => onProgress?.Invoke(7, totalSteps, "PyTorch (CUDA) をインストール中...", detail));
            }
            else if (_detectedDevice == GpuDetectionService.DeviceDirectMl)
            {
                onProgress?.Invoke(7, totalSteps, "PyTorch (DirectML) をインストール中...", null);
                await InstallPackagesAsync(
                    [TorchDirectMlSpec],
                    detail => onProgress?.Invoke(7, totalSteps, "PyTorch (DirectML) をインストール中...", detail));
            }
            else
            {
                onProgress?.Invoke(7, totalSteps, "PyTorch (CPU) をインストール中...", null);
                await InstallPackagesWithIndexAsync(
                    ["torch", "torchvision"],
                    "https://download.pytorch.org/whl/cpu",
                    detail => onProgress?.Invoke(7, totalSteps, "PyTorch (CPU) をインストール中...", detail));
            }

            // 8: GPU拡張ライブラリ (デバイス種別に応じてインストール)
            if (_detectedDevice == GpuDetectionService.DeviceCuda)
            {
                onProgress?.Invoke(8, totalSteps, "xformers をインストール中...", null);
                await InstallPackagesAsync(
                    ["xformers"],
                    detail => onProgress?.Invoke(8, totalSteps, "xformers をインストール中...", detail));
            }
            else if (_detectedDevice == GpuDetectionService.DeviceDirectMl)
            {
                onProgress?.Invoke(8, totalSteps, "DirectML を検証中...", null);
                await EnsureTorchDirectMlRuntimeAsync();
            }
            else
            {
                onProgress?.Invoke(8, totalSteps, "GPU拡張のインストールをスキップ (CPUモード)", null);
            }

            // 9: AI関連ライブラリ
            onProgress?.Invoke(9, totalSteps, "AI関連ライブラリをインストール中...", null);
            await InstallPackagesAsync(
                [
                    "diffusers", "transformers", "accelerate", "safetensors",
                    "Pillow", "pillow-heif", "numpy", "einops", "pytorch_lightning", "open_clip_torch"
                ],
                detail => onProgress?.Invoke(9, totalSteps, "AI関連ライブラリをインストール中...", detail));

            // 10: SUPIRリポジトリをクローン
            onProgress?.Invoke(10, totalSteps, "SUPIR リポジトリを取得中...", null);
            await CloneSupirRepositoryAsync(
                detail => onProgress?.Invoke(10, totalSteps, "SUPIR リポジトリを取得中...", detail));

            // 10.5: SUPIRリポジトリに非CUDA対応パッチを適用
            PatchSupirRepository();

            // 11: SUPIR追加依存パッケージ (ホイールが確実にあるもの)
            onProgress?.Invoke(11, totalSteps, "SUPIR 追加依存をインストール中 (1/2)...", null);
            await InstallPackagesAsync(
                [
                    "omegaconf", "kornia", "opencv-python-headless", "tqdm",
                    "scipy", "timm", "sentencepiece", "tokenizers", "PyYAML"
                ],
                detail => onProgress?.Invoke(11, totalSteps, "SUPIR 追加依存をインストール中 (1/2)...", detail));

            // 12: SUPIR追加依存パッケージ (ソースビルドが必要な場合があるもの)
            // setuptools: openai-clipがpkg_resourcesに依存するため、後続インストールによる
            // 上書き・欠損を防ぐためここでも明示的にインストールする
            onProgress?.Invoke(12, totalSteps, "SUPIR 追加依存をインストール中 (2/2)...", null);
            await InstallPackagesAsync(
                ["facexlib", "k-diffusion", "einops-exts", "openai-clip", SetuptoolsCompatSpec],
                detail => onProgress?.Invoke(12, totalSteps, "SUPIR 追加依存をインストール中 (2/2)...", detail));

            // 13: gradio
            onProgress?.Invoke(13, totalSteps, "Gradio をインストール中...", null);
            await InstallPackagesAsync(
                ["gradio"],
                detail => onProgress?.Invoke(13, totalSteps, "Gradio をインストール中...", detail));

            // 14: SUPIRモデル (チェックポイント) と設定生成
            onProgress?.Invoke(14, totalSteps, "SUPIRモデルを準備中...", null);
            await EnsureSupirModelAssetsAsync(
                detail => onProgress?.Invoke(14, totalSteps, "SUPIRモデルを準備中...", detail));

            // 15: pip自体の更新 + setuptools整合性チェック
            onProgress?.Invoke(15, totalSteps, "pip を更新中...", null);
            await UpgradePipAsync();
            onProgress?.Invoke(15, totalSteps, "setuptools の整合性を確認中...", null);
            await EnsureSetuptoolsRuntimeAsync();

            // 16: マーカー書込 + 完了
            WriteDepsMarker();
            onProgress?.Invoke(totalSteps, totalSteps, "セットアップ完了", null);
            Log("SUPIR依存パッケージのインストールが完了しました。", LogLevel.Info);
            return true;
        }
        catch (Exception ex)
        {
            Log($"初期化中にエラーが発生しました: {ex.Message}", LogLevel.Error);
            LogException("初期化エラー", ex);
            return false;
        }
    }

    /// <summary>
    /// 依存パッケージがインストール済みかチェック
    /// </summary>
    private bool IsDepsInstalled()
    {
        return File.Exists(DepsMarkerFile) && File.ReadAllText(DepsMarkerFile).Trim() == CurrentMarkerVersion;
    }

    /// <summary>
    /// マーカーファイルを書き込み
    /// </summary>
    private void WriteDepsMarker()
    {
        File.WriteAllText(DepsMarkerFile, CurrentMarkerVersion);
    }

    /// <summary>
    /// GPUデバイス種別が前回のインストールから変更されたかチェック
    /// </summary>
    /// <returns>デバイス種別が変更された場合true</returns>
    private bool IsDeviceChanged()
    {
        if (!File.Exists(DepsMarkerFile)) return false;

        var previous = File.ReadAllText(DepsMarkerFile).Trim();

        // 現在のマーカーと一致すれば変更なし
        if (previous == CurrentMarkerVersion) return false;

        // "installed_v{N}_{device}" 形式ならデバイス差分のみを見る
        const string markerPrefix = "installed_v";
        if (previous.StartsWith(markerPrefix, StringComparison.Ordinal))
        {
            var lastUnderscore = previous.LastIndexOf('_');
            if (lastUnderscore >= 0 && lastUnderscore < previous.Length - 1)
            {
                var previousDevice = previous[(lastUnderscore + 1)..];
                return previousDevice != _detectedDevice;
            }
        }

        // 旧形式 (デバイス種別なし) は常にCUDAパッケージを使用していた
        return _detectedDevice != GpuDetectionService.DeviceCuda;
    }

    /// <summary>
    /// デバイス変更時にPyTorch関連パッケージをsite-packagesからクリーンアップする
    /// CUDA版/CPU版/DirectML版の競合を防ぐ
    /// </summary>
    private void CleanupPyTorchPackages()
    {
        if (!Directory.Exists(SitePackagesPath)) return;

        // 削除対象のパッケージディレクトリ名
        string[] targetPackages = ["torch", "torchvision", "xformers", "torch_directml", "triton"];

        foreach (var pkg in targetPackages)
        {
            var pkgDir = Path.Combine(SitePackagesPath, pkg);
            if (Directory.Exists(pkgDir))
            {
                Log($"クリーンアップ: {pkg}/", LogLevel.Info);
                try { Directory.Delete(pkgDir, true); }
                catch (Exception ex) { Log($"クリーンアップ失敗 ({pkg}): {ex.Message}", LogLevel.Warning); }
            }
        }

        // torch-directml のネイティブ拡張 (.pyd) はディレクトリ外に置かれるため個別に削除
        try
        {
            foreach (var file in Directory.GetFiles(SitePackagesPath, "torch_directml_native*.pyd"))
            {
                var name = Path.GetFileName(file);
                Log($"クリーンアップ: {name}", LogLevel.Debug);
                try { File.Delete(file); }
                catch (Exception ex) { Log($"クリーンアップ失敗 ({name}): {ex.Message}", LogLevel.Warning); }
            }
        }
        catch (Exception ex)
        {
            Log($"ファイル列挙中にエラー: {ex.Message}", LogLevel.Warning);
        }

        // .dist-info ディレクトリと NVIDIA CUDA ライブラリを削除
        try
        {
            foreach (var dir in Directory.GetDirectories(SitePackagesPath))
            {
                var name = Path.GetFileName(dir);
                var nameLower = name.ToLowerInvariant();

                // PyTorchパッケージの dist-info
                var isDistInfo = targetPackages.Any(pkg =>
                    nameLower.StartsWith($"{pkg.Replace('-', '_')}-") && nameLower.EndsWith(".dist-info"));

                // NVIDIA CUDA ライブラリ (CUDA torch に付随する nvidia_* パッケージ)
                var isNvidia = nameLower.StartsWith("nvidia");

                if (isDistInfo || isNvidia)
                {
                    Log($"クリーンアップ: {name}/", LogLevel.Debug);
                    try { Directory.Delete(dir, true); }
                    catch (Exception ex) { Log($"クリーンアップ失敗 ({name}): {ex.Message}", LogLevel.Warning); }
                }
            }
        }
        catch (Exception ex)
        {
            Log($"ディレクトリ列挙中にエラー: {ex.Message}", LogLevel.Warning);
        }
    }

    /// <summary>
    /// 起動時にライブラリの最新バージョンをチェックし、更新があるか判定
    /// pip list --outdated で更新可能なパッケージを確認
    /// </summary>
    private async Task<bool> CheckForUpdatesAsync()
    {
        try
        {
            Log("ライブラリの更新確認を開始...", LogLevel.Info);
            // --target は pip install 専用オプション。pip list --outdated では
            // --path でインストール先を指定する (B2-BUG-13: --target は無効)
            var result = await RunPythonCommandAsync(
                $"-m pip list --outdated --format=columns --path \"{SitePackagesPath}\"");

            if (result.ExitCode != 0)
            {
                Log("更新確認に失敗しました。スキップします。", LogLevel.Warning);
                return false;
            }

            // 出力に行がある (ヘッダー2行を除く) = 更新パッケージがある
            var lines = result.StandardOutput.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var outdatedCount = Math.Max(0, lines.Length - 2);

            if (outdatedCount > 0)
            {
                Log($"更新可能なパッケージが {outdatedCount} 個見つかりました。", LogLevel.Info);
                foreach (var line in lines.Skip(2))
                {
                    Log($"  更新: {line.Trim()}", LogLevel.Debug);
                }
                return true;
            }

            Log("すべてのライブラリは最新です。", LogLevel.Info);
            return false;
        }
        catch (Exception ex)
        {
            Log($"更新確認中にエラー: {ex.Message}", LogLevel.Warning);
            return false;
        }
    }

    /// <summary>
    /// 埋め込みPythonの準備を確認し、必要に応じてダウンロードする
    /// </summary>
    private async Task<bool> EnsureEmbeddedPythonAsync()
    {
        if (IsEmbeddedPythonReady(_pythonEmbedDir))
        {
            Log($"既存の埋め込みPythonを検出: {_pythonEmbedDir}", LogLevel.Info);
            _settings.PythonHome = _pythonEmbedDir;
            await CheckPythonVersionUpdateAsync();
            return true;
        }

        return await DownloadEmbeddedPythonAsync();
    }

    /// <summary>
    /// Pythonの新しいマイナーバージョンがあるかチェック
    /// </summary>
    private async Task CheckPythonVersionUpdateAsync()
    {
        try
        {
            var versionFile = Path.Combine(_pythonEmbedDir, "python-embed-version.txt");
            if (!File.Exists(versionFile)) return;

            var currentVersionStr = File.ReadAllText(versionFile).Trim();
            if (!Version.TryParse(currentVersionStr, out var currentVersion)) return;

            var latestVersionStr = await ResolveLatestPythonVersionAsync(skipSaved: true);
            if (string.IsNullOrEmpty(latestVersionStr)) return;
            if (!Version.TryParse(latestVersionStr, out var latestVersion)) return;

            if (latestVersion > currentVersion)
            {
                Log($"新しいPythonバージョンが利用可能: {currentVersion} -> {latestVersion}", LogLevel.Info);
                Log("次回起動時にPythonを更新するためマーカーを削除します。", LogLevel.Info);

                if (File.Exists(DepsMarkerFile))
                    File.Delete(DepsMarkerFile);

                if (Directory.Exists(_pythonEmbedDir))
                    Directory.Delete(_pythonEmbedDir, true);

                await DownloadEmbeddedPythonAsync();
            }
            else
            {
                Log($"Pythonは最新 ({currentVersion})", LogLevel.Debug);
            }
        }
        catch (Exception ex)
        {
            Log($"Pythonバージョンチェック中にエラー: {ex.Message}", LogLevel.Warning);
        }
    }

    /// <summary>
    /// 埋め込みPythonをダウンロードして展開する
    /// </summary>
    private async Task<bool> DownloadEmbeddedPythonAsync()
    {
        try
        {
            var targetVersion = await ResolveLatestPythonVersionAsync();
            if (string.IsNullOrEmpty(targetVersion))
            {
                Log("Pythonバージョンの解決に失敗しました。", LogLevel.Error);
                return false;
            }

            var zipFileName = $"python-{targetVersion}-embed-amd64.zip";
            var zipPath = Path.Combine(_basePythonDir, zipFileName);
            var downloadUrl = $"https://www.python.org/ftp/python/{targetVersion}/{zipFileName}";

            Directory.CreateDirectory(_basePythonDir);
            if (Directory.Exists(_pythonEmbedDir))
                Directory.Delete(_pythonEmbedDir, true);
            Directory.CreateDirectory(_pythonEmbedDir);

            if (!File.Exists(zipPath))
            {
                Log($"埋め込みPythonをダウンロード中: {downloadUrl}", LogLevel.Info);
                using var response = await SharedHttpClient.GetAsync(downloadUrl, HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();
                await using var contentStream = await response.Content.ReadAsStreamAsync();
                await using var fileStream = new FileStream(zipPath, FileMode.Create, FileAccess.Write, FileShare.None);
                await contentStream.CopyToAsync(fileStream);
            }

            Log("埋め込みPythonを展開中...", LogLevel.Info);
            ZipFile.ExtractToDirectory(zipPath, _pythonEmbedDir, true);

            if (!IsEmbeddedPythonReady(_pythonEmbedDir))
            {
                Log("展開後のPython検証に失敗しました。", LogLevel.Error);
                return false;
            }

            WriteVersionFile(targetVersion);
            _settings.PythonVersion = targetVersion;
            _settings.PythonHome = _pythonEmbedDir;
            _settings.Save();

            try { if (File.Exists(zipPath)) File.Delete(zipPath); } catch { /* ignore */ }

            Log($"埋め込みPythonの準備完了: {_pythonEmbedDir}", LogLevel.Info);
            return true;
        }
        catch (Exception ex)
        {
            Log($"埋め込みPythonのダウンロード/展開に失敗: {ex.Message}", LogLevel.Error);
            return false;
        }
    }

    /// <summary>
    /// 利用可能な最新のPythonバージョンを解決する
    /// </summary>
    /// <param name="skipSaved">保存済みバージョンをスキップするか</param>
    private async Task<string?> ResolveLatestPythonVersionAsync(bool skipSaved = false)
    {
        if (!skipSaved && !string.IsNullOrEmpty(_settings.PythonVersion))
        {
            if (Version.TryParse(_settings.PythonVersion, out var v) && v <= MaxEmbeddedPythonVersion)
            {
                Log($"保存済みPythonバージョンを使用: {_settings.PythonVersion}", LogLevel.Info);
                return _settings.PythonVersion;
            }
        }

        try
        {
            Log("python.org から最新バージョンを取得中...", LogLevel.Info);
            var indexContent = await SharedHttpClient.GetStringAsync("https://www.python.org/ftp/python/");

            var versions = new List<Version>();
            foreach (Match match in VersionRegex().Matches(indexContent))
            {
                if (Version.TryParse(match.Groups["version"].Value, out var version))
                    versions.Add(version);
            }

            versions.Sort((a, b) => b.CompareTo(a));

            foreach (var version in versions)
            {
                if (version > MaxEmbeddedPythonVersion) continue;

                var zipFileName = $"python-{version}-embed-amd64.zip";
                var url = $"https://www.python.org/ftp/python/{version}/{zipFileName}";

                try
                {
                    using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                    using var request = new HttpRequestMessage(HttpMethod.Head, url);
                    using var response = await SharedHttpClient.SendAsync(request, cts.Token);
                    if (response.IsSuccessStatusCode)
                    {
                        Log($"ダウンロード可能なバージョン: {version}", LogLevel.Info);
                        return version.ToString();
                    }
                }
                catch { /* try next */ }
            }
        }
        catch (Exception ex)
        {
            Log($"バージョン取得に失敗: {ex.Message}", LogLevel.Error);
        }

        return null;
    }

    /// <summary>
    /// Python ._pthファイルを設定する
    /// </summary>
    private void ConfigurePythonPth()
    {
        var pthFiles = Directory.GetFiles(_pythonEmbedDir, "python*._pth");
        if (pthFiles.Length == 0) return;

        // 複数ファイルがある場合は警告し、アルファベット順で最初のものを選択 (B2-BUG-08)
        if (pthFiles.Length > 1)
        {
            Log($"複数の ._pth ファイルが見つかりました ({pthFiles.Length} 個)。最初のファイルを使用します: {string.Join(", ", pthFiles.Select(Path.GetFileName))}", LogLevel.Warning);
            Array.Sort(pthFiles, StringComparer.OrdinalIgnoreCase);
        }
        var pthFile = pthFiles[0];
        var content = File.ReadAllText(pthFile);

        if (content.Contains("#import site"))
            content = content.Replace("#import site", "import site");
        else if (!content.Contains("import site"))
            content += "\r\nimport site";

        if (!content.Contains("site-packages", StringComparison.OrdinalIgnoreCase))
            content += $"\r\n{SitePackagesPath}";

        // SUPIRリポジトリパスを追加 (埋め込みPythonはPYTHONPATH環境変数を無視するため)
        if (!content.Contains(SupirRepoPath, StringComparison.OrdinalIgnoreCase))
            content += $"\r\n{SupirRepoPath}";

        File.WriteAllText(pthFile, content);
        Log("._pth設定を更新しました。", LogLevel.Debug);
    }

    /// <summary>
    /// SUPIRリポジトリをクローンまたは更新する
    /// </summary>
    /// <param name="onDetail">詳細情報のコールバック</param>
    private async Task CloneSupirRepositoryAsync(Action<string>? onDetail = null)
    {
        // 既にクローン済みの場合はpullで更新
        if (Directory.Exists(SupirRepoPath) && Directory.Exists(Path.Combine(SupirRepoPath, "SUPIR")))
        {
            Log($"SUPIRリポジトリは既にクローン済み: {SupirRepoPath}", LogLevel.Info);
            onDetail?.Invoke("既存のリポジトリを検出");

            try
            {
                onDetail?.Invoke("最新コードを取得中...");
                await RunCommandAsync("git", $"-C \"{SupirRepoPath}\" pull --ff-only");
                Log("SUPIRリポジトリを更新しました。", LogLevel.Info);
            }
            catch (Exception ex)
            {
                Log($"git pull に失敗しましたが、既存のリポジトリを使用します: {ex.Message}", LogLevel.Warning);
            }

            return;
        }

        // 不完全なクローンがあれば削除
        if (Directory.Exists(SupirRepoPath))
        {
            Log("不完全なSUPIRリポジトリを削除中...", LogLevel.Info);
            Directory.Delete(SupirRepoPath, true);
        }

        Log($"SUPIRリポジトリをクローン中: {SupirRepoPath}", LogLevel.Info);
        onDetail?.Invoke("git clone を実行中...");

        var result = await RunCommandAsync(
            "git",
            $"clone --depth 1 https://github.com/Fanghua-Yu/SUPIR.git \"{SupirRepoPath}\"",
            onOutput: line =>
            {
                // git cloneの進捗表示をパース
                var detail = ParseGitOutputDetail(line);
                if (detail != null)
                    onDetail?.Invoke(detail);
            });

        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"SUPIRリポジトリのクローンに失敗しました。終了コード: {result.ExitCode}");
        }

        Log("SUPIRリポジトリのクローンが完了しました。", LogLevel.Info);
        onDetail?.Invoke("クローン完了");
    }

    /// <summary>
    /// SUPIRリポジトリのソースコードにパッチを当てる。
    /// upstream の sampling.py が device='cuda' をハードコードしているため、
    /// DirectML / CPU 環境でも動作するよう修正する。
    /// </summary>
    private void PatchSupirRepository()
    {
        var samplingPy = Path.Combine(SupirRepoPath, "sgm", "modules", "diffusionmodules", "sampling.py");
        if (!File.Exists(samplingPy))
        {
            Log($"パッチ対象ファイルが見つかりません: {samplingPy}", LogLevel.Warning);
            return;
        }

        var content = File.ReadAllText(samplingPy);
        var patched = false;

        // gaussian_weights の device='cuda' を device='cpu' に変更
        const string cudaDevice = "device='cuda'";
        const string cpuDevice = "device='cpu'";
        if (content.Contains(cudaDevice))
        {
            content = content.Replace(cudaDevice, cpuDevice);
            patched = true;
        }

        // tile_weights の使用箇所で .to(x.device) を挿入
        const string oldRepeat = "self.tile_weights.repeat(b,";
        const string newRepeat = "self.tile_weights.to(x.device).repeat(b,";
        if (content.Contains(oldRepeat) && !content.Contains(newRepeat))
        {
            content = content.Replace(oldRepeat, newRepeat);
            patched = true;
        }

        // repeat の前にスペースがある場合にも対応
        const string oldRepeatSpaced = "self.tile_weights.repeat(b, ";
        const string newRepeatSpaced = "self.tile_weights.to(x.device).repeat(b, ";
        if (content.Contains(oldRepeatSpaced) && !content.Contains(newRepeatSpaced))
        {
            content = content.Replace(oldRepeatSpaced, newRepeatSpaced);
            patched = true;
        }

        if (patched)
        {
            File.WriteAllText(samplingPy, content);
            Log("SUPIRリポジトリにパッチを適用しました (sampling.py: device='cuda' → device='cpu', tile_weights デバイス移動追加)", LogLevel.Info);
        }
        else
        {
            Log("SUPIRリポジトリのパッチは既に適用済みです。", LogLevel.Debug);
        }
    }

    /// <summary>
    /// DirectML (torch_directml) を検証し、必要なら修復する
    /// </summary>
    private async Task EnsureTorchDirectMlRuntimeAsync()
    {
        if (await VerifyTorchDirectMlAsync())
        {
            Log("DirectML の検証に成功しました。", LogLevel.Debug);
            return;
        }

        Log("DirectML の検証に失敗したため、PyTorch/DirectML を修復します。", LogLevel.Warning);
        CleanupPyTorchPackages();

        await InstallPackagesAsync([TorchDirectMlSpec], detail => Log(detail, LogLevel.Debug));

        if (!await VerifyTorchDirectMlAsync())
        {
            throw new InvalidOperationException(
                "DirectML の初期化に失敗しました。GPUドライバと PyTorch/torch-directml の互換性を確認してください。");
        }

        Log("DirectML を修復し、検証に成功しました。", LogLevel.Info);
    }

    /// <summary>
    /// DirectML が利用可能かを簡易チェックする
    /// </summary>
    /// <returns>DirectML が利用可能なら true</returns>
    private async Task<bool> VerifyTorchDirectMlAsync()
    {
        try
        {
            var versions = await RunPythonCommandAsync(
                "-c \"import torch, torchvision; print(torch.__version__); print(torchvision.__version__)\"");
            if (versions.ExitCode == 0)
            {
                var torchVer = GetLastNonEmptyLine(versions.StandardOutput) ?? "";
                var lines = versions.StandardOutput
                    .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
                    .Select(l => l.Trim())
                    .ToArray();

                if (lines.Length >= 2)
                {
                    Log($"PyTorch: {lines[0]}, torchvision: {lines[1]}", LogLevel.Debug);
                }
                else if (!string.IsNullOrWhiteSpace(torchVer))
                {
                    Log($"PyTorch: {torchVer}", LogLevel.Debug);
                }
            }

            var importResult = await RunPythonCommandAsync("-c \"import torch_directml\"");
            if (importResult.ExitCode != 0)
            {
                var details = string.IsNullOrWhiteSpace(importResult.StandardError)
                    ? importResult.StandardOutput.Trim()
                    : importResult.StandardError.Trim();
                Log($"torch-directml import failed: {details}", LogLevel.Warning);
                return false;
            }

            var result = await RunPythonCommandAsync(
                "-c \"import torch; import torch_directml; dml=torch_directml.device(); a=torch.tensor([1]).to(dml); b=torch.tensor([2]).to(dml); print((a+b).item())\"");

            var lastLine = GetLastNonEmptyLine(result.StandardOutput);
            if (result.ExitCode == 0 && lastLine == "3")
            {
                return true;
            }

            var failureDetails = string.IsNullOrWhiteSpace(result.StandardError)
                ? result.StandardOutput.Trim()
                : result.StandardError.Trim();
            Log($"DirectML 検証失敗: {failureDetails}", LogLevel.Warning);
            return false;
        }
        catch (Exception ex)
        {
            Log($"DirectML 検証中に例外が発生しました: {ex.Message}", LogLevel.Warning);
            return false;
        }
    }

    /// <summary>
    /// HEIC保存用の pillow-heif が利用可能か確認し、未導入ならインストールを試みる
    /// </summary>
    private async Task EnsurePillowHeifAsync()
    {
        try
        {
            var check = await RunPythonCommandAsync("-c \"import pillow_heif\"");
            if (check.ExitCode == 0) return;

            Log("pillow-heif が見つからないためインストールを試みます。", LogLevel.Warning);
            await InstallPackagesAsync(["pillow-heif"]);

            var recheck = await RunPythonCommandAsync("-c \"import pillow_heif\"");
            if (recheck.ExitCode == 0) return;

            var details = string.IsNullOrWhiteSpace(recheck.StandardError)
                ? recheck.StandardOutput.Trim()
                : recheck.StandardError.Trim();
            Log($"pillow-heif の検証に失敗しました (HEIC保存不可): {details}", LogLevel.Warning);
        }
        catch (Exception ex)
        {
            Log($"pillow-heif の検証に失敗しました (HEIC保存不可): {ex.Message}", LogLevel.Warning);
        }
    }

    /// <summary>
    /// SUPIRのチェックポイントを取得し、GUI用設定ファイルを生成する
    /// </summary>
    /// <param name="onDetail">詳細表示用のコールバック</param>
    private async Task EnsureSupirModelAssetsAsync(Action<string>? onDetail = null)
    {
        Directory.CreateDirectory(SupirConfigDir);

        // CKPT_PTH.py は upstream で Linux 固定パスになっているため、
        // site-packages に上書き用を置いて Hugging Face から自動取得させる。
        WriteCkptPthOverrideFile();

        // huggingface_hub が無い場合は追加インストール
        var hubCheck = await RunPythonCommandAsync("-c \"import huggingface_hub\"");
        if (hubCheck.ExitCode != 0)
        {
            onDetail?.Invoke("huggingface-hub をインストール中...");
            await InstallPackagesAsync(["huggingface-hub"], onDetail);
        }

        onDetail?.Invoke("チェックポイントを確認中...");
        var sdxlCkptPath = await DownloadFromHuggingFaceAsync(SdxlBaseCkptFileName, onDetail);
        var supirQCkptPath = await DownloadFromHuggingFaceAsync(SupirV0QCkptFileName, onDetail);

        // GUIでは v0-Q を使用するため v0-F は未取得のまま (~) にする
        string? supirFCkptPath = null;

        onDetail?.Invoke("SUPIR設定ファイルを生成中...");
        var sourceV0 = Path.Combine(SupirRepoPath, "options", "SUPIR_v0.yaml");
        var sourceV0Tiled = Path.Combine(SupirRepoPath, "options", "SUPIR_v0_tiled.yaml");

        WriteSupirConfigFile(sourceV0, SupirConfigV0Path, sdxlCkptPath, supirQCkptPath, supirFCkptPath);
        WriteSupirConfigFile(sourceV0Tiled, SupirConfigV0TiledPath, sdxlCkptPath, supirQCkptPath, supirFCkptPath);
    }

    /// <summary>
    /// GUI用設定ファイルが無い場合のみ、SUPIRモデル資産を準備する
    /// </summary>
    private async Task EnsureSupirModelAssetsIfMissingAsync()
    {
        if (File.Exists(SupirConfigV0Path) && File.Exists(SupirConfigV0TiledPath))
            return;

        if (!Directory.Exists(SupirRepoPath) || !File.Exists(Path.Combine(SupirRepoPath, "options", "SUPIR_v0.yaml")))
        {
            await CloneSupirRepositoryAsync();
            PatchSupirRepository();
        }

        await EnsureSupirModelAssetsAsync();
    }

    /// <summary>
    /// Hugging Face Hub からチェックポイントを取得する (キャッシュ済みなら即時)
    /// </summary>
    /// <param name="fileName">取得するファイル名</param>
    /// <param name="onDetail">詳細表示用のコールバック</param>
    private async Task<string> DownloadFromHuggingFaceAsync(string fileName, Action<string>? onDetail = null)
    {
        onDetail?.Invoke($"ダウンロード: {fileName}");

        const string prefix = "SUPIR_GUI_HF_PATH:";
        var args =
            "-c \"from huggingface_hub import hf_hub_download; " +
            $"p=hf_hub_download(repo_id='{SupirWeightsRepoId}', filename='{fileName}'); " +
            $"print('{prefix}'+p)\"";

        var result = await RunPythonCommandAsync(args);
        if (result.ExitCode != 0)
        {
            var details = string.IsNullOrWhiteSpace(result.StandardError)
                ? result.StandardOutput.Trim()
                : result.StandardError.Trim();
            throw new InvalidOperationException($"チェックポイントの取得に失敗しました ({fileName}): {details}");
        }

        var path = FindPrefixedValue(result.StandardOutput, prefix) ?? "";
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            throw new InvalidOperationException($"チェックポイントの取得結果パスが不正です ({fileName}): {path}");
        }

        Log($"チェックポイント取得完了: {fileName} -> {path}", LogLevel.Info);
        return path;
    }

    /// <summary>
    /// SUPIRの upstream CKPT_PTH.py を上書きする (Hugging Face 自動取得用)
    /// </summary>
    private void WriteCkptPthOverrideFile()
    {
        Directory.CreateDirectory(SitePackagesPath);
        var path = Path.Combine(SitePackagesPath, "CKPT_PTH.py");

        const string content =
            "LLAVA_CLIP_PATH = None\r\n" +
            "LLAVA_MODEL_PATH = None\r\n" +
            "SDXL_CLIP1_PATH = None\r\n" +
            "SDXL_CLIP2_CKPT_PTH = None\r\n";

        File.WriteAllText(path, content, Encoding.UTF8);
        Log("CKPT_PTH.py を上書き生成しました。", LogLevel.Debug);
    }

    /// <summary>
    /// upstream のSUPIR設定(YAML)を元に、チェックポイントパスのみ差し替えた設定ファイルを生成する
    /// </summary>
    private void WriteSupirConfigFile(
        string sourcePath,
        string destinationPath,
        string sdxlCkptPath,
        string supirQCkptPath,
        string? supirFCkptPath)
    {
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("SUPIR設定ファイルが見つかりません。", sourcePath);

        var yaml = File.ReadAllText(sourcePath, Encoding.UTF8);

        yaml = ReplaceYamlRootValue(yaml, "SDXL_CKPT", ToYamlSingleQuoted(sdxlCkptPath));
        yaml = ReplaceYamlRootValue(yaml, "SUPIR_CKPT_Q", ToYamlSingleQuoted(supirQCkptPath));

        var fValue = supirFCkptPath != null ? ToYamlSingleQuoted(supirFCkptPath) : "~";
        yaml = ReplaceYamlRootValue(yaml, "SUPIR_CKPT_F", fValue);

        // 出力はCRLFに統一
        yaml = yaml.Replace("\r\n", "\n", StringComparison.Ordinal).Replace("\n", "\r\n", StringComparison.Ordinal);

        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath) ?? SupirConfigDir);
        File.WriteAllText(destinationPath, yaml, Encoding.UTF8);
        Log($"SUPIR設定ファイル生成: {destinationPath}", LogLevel.Info);
    }

    /// <summary>
    /// YAMLルートレベルのキー値行を置換する
    /// </summary>
    private static string ReplaceYamlRootValue(string yaml, string key, string value)
    {
        var pattern = $@"^(?<indent>\s*){Regex.Escape(key)}\s*:\s*.*$";
        var replaced = Regex.Replace(
            yaml,
            pattern,
            $"${{indent}}{key}: {value}",
            RegexOptions.Multiline);

        if (ReferenceEquals(replaced, yaml))
            throw new InvalidOperationException($"YAMLキーが見つかりません: {key}");

        return replaced;
    }

    /// <summary>
    /// YAMLで安全に扱える単一引用符文字列へ変換する
    /// </summary>
    private static string ToYamlSingleQuoted(string value)
    {
        return $"'{value.Replace("'", "''", StringComparison.Ordinal)}'";
    }

    /// <summary>
    /// 文字列から最後の非空行を取得する
    /// </summary>
    private static string? GetLastNonEmptyLine(string text)
    {
        var lines = text.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        return lines.Length == 0 ? null : lines[^1].Trim();
    }

    /// <summary>
    /// テキスト内の "prefix + value" 行から value を取得する
    /// </summary>
    private static string? FindPrefixedValue(string text, string prefix)
    {
        foreach (var line in text.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            var trimmed = line.Trim();
            if (trimmed.StartsWith(prefix, StringComparison.Ordinal))
                return trimmed[prefix.Length..].Trim();
        }

        return null;
    }

    /// <summary>
    /// pipが利用可能かチェック
    /// </summary>
    private async Task<bool> IsPipAvailableAsync()
    {
        var result = await RunPythonCommandAsync("-m pip --version");
        return result.ExitCode == 0;
    }

    /// <summary>
    /// pipをインストールする
    /// </summary>
    private async Task InstallPipAsync()
    {
        Log("pip をインストール中...", LogLevel.Info);
        var getPipPath = Path.Combine(_pythonEmbedDir, "get-pip.py");

        using (var response = await SharedHttpClient.GetAsync("https://bootstrap.pypa.io/get-pip.py"))
        {
            response.EnsureSuccessStatusCode();
            await File.WriteAllBytesAsync(getPipPath, await response.Content.ReadAsByteArrayAsync());
        }

        var result = await RunPythonCommandAsync(
            $"\"{getPipPath}\" --disable-pip-version-check --no-warn-script-location -v --target \"{SitePackagesPath}\"");

        if (result.ExitCode != 0)
        {
            Log($"pip インストール失敗: {result.StandardError}", LogLevel.Error);
            throw new InvalidOperationException($"pip のインストールに失敗しました。終了コード: {result.ExitCode}");
        }

        if (File.Exists(getPipPath))
            File.Delete(getPipPath);

        Log("pip のインストールが完了しました。", LogLevel.Info);
    }

    /// <summary>
    /// pipを最新バージョンに更新する
    /// </summary>
    private async Task UpgradePipAsync()
    {
        try
        {
            var result = await RunPythonCommandAsync(
                $"-m pip install --upgrade pip --target \"{SitePackagesPath}\" --no-warn-script-location");
            if (result.ExitCode == 0)
                Log("pip を最新バージョンに更新しました。", LogLevel.Debug);
        }
        catch (Exception ex)
        {
            Log($"pip更新をスキップ: {ex.Message}", LogLevel.Warning);
        }
    }

    /// <summary>
    /// setuptools が pkg_resources を提供していることを検証し、欠損時は再インストールする
    /// </summary>
    private async Task EnsureSetuptoolsRuntimeAsync()
    {
        const string verificationArgs = "-c \"import pkg_resources; import setuptools\"";
        var verifyResult = await RunPythonCommandAsync(verificationArgs);
        if (verifyResult.ExitCode == 0)
        {
            Log("setuptools / pkg_resources の整合性確認に成功しました。", LogLevel.Debug);
            return;
        }

        Log("pkg_resources が見つからないため setuptools を再インストールします。", LogLevel.Warning);
        var reinstallResult = await RunPythonCommandAsync(
            $"-m pip install --upgrade --force-reinstall --no-deps --no-warn-script-location --target \"{SitePackagesPath}\" \"{SetuptoolsCompatSpec}\"");
        if (reinstallResult.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"setuptools の再インストールに失敗しました。終了コード: {reinstallResult.ExitCode}");
        }

        var reverifyResult = await RunPythonCommandAsync(verificationArgs);
        if (reverifyResult.ExitCode != 0)
        {
            throw new InvalidOperationException(
                "setuptools の再インストール後も pkg_resources を読み込めません。");
        }

        Log("setuptools を再インストールし、pkg_resources を復旧しました。", LogLevel.Info);
    }

    /// <summary>
    /// パッケージをインストールする
    /// </summary>
    /// <param name="packages">インストールするパッケージのリスト</param>
    /// <param name="onDetail">詳細進捗のコールバック</param>
    private async Task InstallPackagesAsync(IEnumerable<string> packages, Action<string>? onDetail = null)
    {
        var packageList = string.Join(" ", packages);
        Log($"パッケージをインストール中: {packageList}", LogLevel.Info);

        // pip出力をパースして詳細情報を通知するハンドラ
        Action<string>? outputHandler = onDetail != null
            ? line =>
            {
                var detail = ParsePipOutputDetail(line);
                if (detail != null)
                    onDetail(detail);
            }
            : null;

        var args = $"-m pip install --upgrade --upgrade-strategy only-if-needed --prefer-binary -v --no-warn-script-location --target \"{SitePackagesPath}\" {packageList}";
        var result = await RunPythonCommandAsync(args, outputHandler);

        if (result.ExitCode != 0)
        {
            Log($"パッケージインストール失敗 ({packageList}): {result.StandardError}", LogLevel.Error);
            throw new InvalidOperationException($"パッケージのインストールに失敗しました ({packageList})。終了コード: {result.ExitCode}");
        }
    }

    /// <summary>
    /// カスタムインデックスURLを指定してパッケージをインストールする
    /// </summary>
    /// <param name="packages">インストールするパッケージのリスト</param>
    /// <param name="indexUrl">pipのインデックスURL</param>
    /// <param name="onDetail">詳細進捗のコールバック</param>
    private async Task InstallPackagesWithIndexAsync(IEnumerable<string> packages, string indexUrl, Action<string>? onDetail = null)
    {
        var packageList = string.Join(" ", packages);
        Log($"パッケージをインストール中 (index: {indexUrl}): {packageList}", LogLevel.Info);

        Action<string>? outputHandler = onDetail != null
            ? line =>
            {
                var detail = ParsePipOutputDetail(line);
                if (detail != null)
                    onDetail(detail);
            }
            : null;

        var args = $"-m pip install --upgrade --upgrade-strategy only-if-needed --prefer-binary -v --no-warn-script-location --target \"{SitePackagesPath}\" --index-url {indexUrl} {packageList}";
        var result = await RunPythonCommandAsync(args, outputHandler);

        if (result.ExitCode != 0)
        {
            Log($"パッケージインストール失敗 ({packageList}): {result.StandardError}", LogLevel.Error);
            throw new InvalidOperationException($"パッケージのインストールに失敗しました ({packageList})。終了コード: {result.ExitCode}");
        }
    }

    /// <summary>
    /// pip出力行から詳細情報をパースする
    /// </summary>
    /// <param name="line">pip出力の1行</param>
    /// <returns>パースされた詳細情報、該当しない場合はnull</returns>
    private static string? ParsePipOutputDetail(string line)
    {
        var trimmed = line.TrimStart();

        var collectingMatch = PipCollectingRegex().Match(trimmed);
        if (collectingMatch.Success)
            return $"確認中: {collectingMatch.Groups[1].Value}";

        var downloadingMatch = PipDownloadingRegex().Match(trimmed);
        if (downloadingMatch.Success)
        {
            var url = downloadingMatch.Groups[1].Value;
            // URLからファイル名を抽出
            var lastSlash = url.LastIndexOf('/');
            var fileName = lastSlash >= 0 ? url[(lastSlash + 1)..] : url;
            return $"ダウンロード中: {fileName}";
        }

        var cachedMatch = PipUsingCachedRegex().Match(trimmed);
        if (cachedMatch.Success)
        {
            var url = cachedMatch.Groups[1].Value;
            var lastSlash = url.LastIndexOf('/');
            var fileName = lastSlash >= 0 ? url[(lastSlash + 1)..] : url;
            return $"キャッシュ使用: {fileName}";
        }

        var installingMatch = PipInstallingRegex().Match(trimmed);
        if (installingMatch.Success)
            return $"書込み中: {installingMatch.Groups[1].Value}";

        var successMatch = PipSuccessRegex().Match(trimmed);
        if (successMatch.Success)
            return "インストール完了";

        return null;
    }

    /// <summary>
    /// git出力行から詳細情報をパースする
    /// </summary>
    /// <param name="line">git出力の1行</param>
    /// <returns>パースされた詳細情報、該当しない場合はnull</returns>
    private static string? ParseGitOutputDetail(string line)
    {
        var trimmed = line.Trim();

        if (trimmed.StartsWith("Cloning into", StringComparison.OrdinalIgnoreCase))
            return "リポジトリをクローン中...";

        if (trimmed.StartsWith("Updating files:", StringComparison.OrdinalIgnoreCase))
            return trimmed;

        if (trimmed.Contains("done.", StringComparison.OrdinalIgnoreCase))
            return "完了";

        return null;
    }

    /// <summary>
    /// Pythonコマンドを実行する
    /// </summary>
    /// <param name="arguments">Pythonに渡す引数</param>
    /// <param name="onOutput">出力行のコールバック</param>
    /// <returns>プロセスの実行結果</returns>
    private async Task<ProcessResult> RunPythonCommandAsync(string arguments, Action<string>? onOutput = null)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = PythonExePath,
            Arguments = arguments,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        ConfigurePythonEnvironment(startInfo);
        return await RunProcessAsync(startInfo, onOutput);
    }

    /// <summary>
    /// 任意のコマンドを実行する
    /// </summary>
    /// <param name="fileName">実行ファイル名</param>
    /// <param name="arguments">引数</param>
    /// <param name="onOutput">出力行のコールバック</param>
    /// <returns>プロセスの実行結果</returns>
    private static async Task<ProcessResult> RunCommandAsync(string fileName, string arguments, Action<string>? onOutput = null)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        return await RunProcessAsync(startInfo, onOutput);
    }

    /// <summary>
    /// プロセスを実行し、結果を返す
    /// </summary>
    /// <param name="startInfo">プロセス起動情報</param>
    /// <param name="onOutput">出力行のコールバック</param>
    /// <returns>プロセスの実行結果</returns>
    private static async Task<ProcessResult> RunProcessAsync(ProcessStartInfo startInfo, Action<string>? onOutput = null)
    {
        using var process = new Process { StartInfo = startInfo };

        if (onOutput != null)
        {
            var stdoutBuilder = new StringBuilder();
            var stderrBuilder = new StringBuilder();
            var outputLock = new object();

            // イベントベースで出力をリアルタイム受信
            process.OutputDataReceived += (_, e) =>
            {
                if (e.Data == null) return;
                lock (outputLock) { stdoutBuilder.AppendLine(e.Data); }
                onOutput(e.Data);
            };

            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data == null) return;
                lock (outputLock) { stderrBuilder.AppendLine(e.Data); }
                onOutput(e.Data);
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();
            // 非同期イベントキューをフラッシュする (B2-BUG-12)
            // WaitForExitAsync() だけでは OutputDataReceived/ErrorDataReceived
            // イベントがすべて届く前に制御が戻る場合がある。
            // 引数なしの WaitForExit() を追加することで確実にフラッシュする。
            process.WaitForExit();

            string stdout, stderr;
            lock (outputLock)
            {
                stdout = stdoutBuilder.ToString();
                stderr = stderrBuilder.ToString();
            }

            return new ProcessResult(process.ExitCode, stdout, stderr);
        }
        else
        {
            process.Start();
            // Read stdout and stderr concurrently to avoid deadlock
            // when OS pipe buffer fills up on one stream
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            var stdout = await stdoutTask;
            var stderr = await stderrTask;
            return new ProcessResult(process.ExitCode, stdout, stderr);
        }
    }

    /// <summary>
    /// Python環境の環境変数を設定する
    /// </summary>
    /// <param name="startInfo">プロセス起動情報</param>
    public void ConfigurePythonEnvironment(ProcessStartInfo startInfo)
    {
        startInfo.Environment["PYTHONHOME"] = _pythonEmbedDir;

        // PYTHONPATHにsite-packagesとSUPIRリポジトリを含める
        var pythonPaths = new List<string> { SitePackagesPath };
        if (Directory.Exists(SupirRepoPath))
            pythonPaths.Add(SupirRepoPath);
        startInfo.Environment["PYTHONPATH"] = string.Join(";", pythonPaths);

        startInfo.Environment["PYTHONUTF8"] = "1";
        startInfo.Environment["HF_HOME"] = Path.Combine(_libBaseDir, "hf_cache");
        startInfo.Environment["HF_HUB_DISABLE_SYMLINKS"] = "1";
        startInfo.Environment["TQDM_MININTERVAL"] = "1";
        startInfo.Environment["SUPIR_GUI_CONFIG_V0"] = SupirConfigV0Path;
        startInfo.Environment["SUPIR_GUI_CONFIG_V0_TILED"] = SupirConfigV0TiledPath;

        var originalPath = Environment.GetEnvironmentVariable("PATH") ?? "";
        startInfo.Environment["PATH"] = $"{_pythonEmbedDir};{originalPath}";
    }

    /// <summary>
    /// 埋め込みPythonが準備できているかチェック
    /// </summary>
    /// <param name="path">チェック対象のパス</param>
    /// <returns>準備できているかどうか</returns>
    private static bool IsEmbeddedPythonReady(string path)
    {
        return Directory.Exists(path)
               && File.Exists(Path.Combine(path, "python.exe"))
               && Directory.GetFiles(path, "python*.dll").Length > 0;
    }

    /// <summary>
    /// Pythonバージョンファイルを書き込む
    /// </summary>
    /// <param name="version">バージョン文字列</param>
    private void WriteVersionFile(string version)
    {
        File.WriteAllText(Path.Combine(_pythonEmbedDir, "python-embed-version.txt"), version);
    }

    /// <summary>
    /// プロセス実行結果
    /// </summary>
    /// <param name="ExitCode">終了コード</param>
    /// <param name="StandardOutput">標準出力</param>
    /// <param name="StandardError">標準エラー出力</param>
    private sealed record ProcessResult(int ExitCode, string StandardOutput, string StandardError);
}
