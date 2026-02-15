using System.Reflection;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SUPIR_GUI.Models;
using SUPIR_GUI.Services;
using static SUPIR_GUI.Services.Logger;

namespace SUPIR_GUI.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    // ========== Settings ==========
    private AppSettings _settings;
    private PythonBootstrapService? _bootstrapService;
    private SupirWrapperService? _supirWrapper;

    /// <summary>アセンブリバージョンから取得した表示用バージョン文字列</summary>
    public static string VersionDisplay { get; } =
        $"v{Assembly.GetExecutingAssembly().GetName().Version?.ToString(3) ?? "0.0.0"}";

    // ========== Initialization State ==========
    [ObservableProperty]
    private bool _isInitializing;

    [ObservableProperty]
    private bool _isInitialized;

    [ObservableProperty]
    private bool _initializationFailed;

    [ObservableProperty]
    private string _initStatusText = "初期化を準備中...";

    [ObservableProperty]
    private double? _initProgress;

    [ObservableProperty]
    private string _initProgressText = "";

    // ========== Tool Download (shown during init) ==========
    /// <summary>ツールダウンロード中フラグ</summary>
    [ObservableProperty]
    private bool _isToolDownloading;

    /// <summary>ツールダウンロードの状態テキスト</summary>
    [ObservableProperty]
    private string _toolDownloadStatus = "";

    /// <summary>ツールダウンロードの進捗率 (0-100、nullで不確定)</summary>
    [ObservableProperty]
    private double? _toolDownloadProgress;

    /// <summary>ツールダウンロードのステップ表示テキスト</summary>
    [ObservableProperty]
    private string _toolDownloadStepText = "";

    /// <summary>ツールダウンロードの詳細情報テキスト</summary>
    [ObservableProperty]
    private string _toolDownloadDetail = "";

    // ========== Processing State ==========
    [ObservableProperty]
    private bool _isProcessing;

    [ObservableProperty]
    private double? _progressPercentage;

    // ========== IO Settings ==========
    [ObservableProperty]
    private IoFormMode _ioFormMode = IoFormMode.FileSelection;

    [ObservableProperty]
    private string _inputFilePath = "";

    [ObservableProperty]
    private string _outputFilePath = "";

    [ObservableProperty]
    private string _inputFolderPath = "";

    [ObservableProperty]
    private string _outputFolderPath = "";

    // ========== SUPIR Parameters ==========
    [ObservableProperty]
    private int _upscaleRatioIndex; // 0=2x, 1=3x, 2=4x

    [ObservableProperty]
    private int _outputFormatIndex; // 0=PNG, 1=JPEG, 2=WebP, 3=HEIC

    /// <summary>EDMステップ数 (固定値)</summary>
    private const int EdmSteps = 50;

    /// <summary>ガイダンススケール (固定値)</summary>
    private const double GuidanceScale = 6.0;

    [ObservableProperty]
    private string _seedText = "";

    [ObservableProperty]
    private bool _useRandomSeed = true;

    /// <summary>s_stage1 推奨固定値</summary>
    private const double SStage1 = -1.0;

    /// <summary>s_stage2 推奨固定値</summary>
    private const double SStage2 = 1.0;

    /// <summary>s_churn 推奨固定値</summary>
    private const double SChurn = 5.0;

    /// <summary>s_noise 推奨固定値</summary>
    private const double SNoise = 1.003;

    /// <summary>タイルサイズ 推奨固定値</summary>
    private const int TiledSize = 512;

    /// <summary>タイリング使用 推奨固定値</summary>
    private const bool UseTiling = true;

    // ========== Computed ==========
    public bool IsSeedEnabled => !UseRandomSeed;

    partial void OnUseRandomSeedChanged(bool value)
    {
        OnPropertyChanged(nameof(IsSeedEnabled));
    }

    // ========== Upscale Ratio Helpers ==========
    private int UpscaleFactor => UpscaleRatioIndex switch
    {
        0 => 2,
        1 => 3,
        2 => 4,
        _ => 2
    };

    private string OutputFormatString => OutputFormatIndex switch
    {
        0 => "png",
        1 => "jpg",
        2 => "webp",
        3 => "heic",
        _ => "png"
    };

    private string OutputExtension => OutputFormatIndex switch
    {
        0 => ".png",
        1 => ".jpg",
        2 => ".webp",
        3 => ".heic",
        _ => ".png"
    };

    // ========== Auto-path generation ==========
    partial void OnInputFilePathChanged(string value)
    {
        if (!string.IsNullOrEmpty(value) && File.Exists(value))
        {
            var dir = Path.GetDirectoryName(value) ?? "";
            var name = Path.GetFileNameWithoutExtension(value);
            OutputFilePath = Path.Combine(dir, $"{name}_supir_{UpscaleFactor}x{OutputExtension}");
        }
    }

    partial void OnInputFolderPathChanged(string value)
    {
        if (!string.IsNullOrEmpty(value) && Directory.Exists(value))
        {
            OutputFolderPath = $"{value}_supir_{UpscaleFactor}x";
        }
    }

    partial void OnUpscaleRatioIndexChanged(int value)
    {
        // Regenerate output paths
        if (!string.IsNullOrEmpty(InputFilePath))
            OnInputFilePathChanged(InputFilePath);
        if (!string.IsNullOrEmpty(InputFolderPath))
            OnInputFolderPathChanged(InputFolderPath);
    }

    partial void OnOutputFormatIndexChanged(int value)
    {
        if (!string.IsNullOrEmpty(InputFilePath))
            OnInputFilePathChanged(InputFilePath);
    }

    // ========== Constructor ==========
    public MainWindowViewModel()
    {
        _settings = AppSettings.Load();
        LoadFromSettings();
    }

    private void LoadFromSettings()
    {
        UseRandomSeed = _settings.UseRandomSeed;
        SeedText = _settings.Seed?.ToString() ?? "";

        UpscaleRatioIndex = _settings.UpscaleFactor switch
        {
            3 => 1,
            4 => 2,
            _ => 0
        };

        OutputFormatIndex = _settings.OutputFormat switch
        {
            "Jpg" => 1,
            "WebP" => 2,
            "Heic" => 3,
            _ => 0
        };
    }

    private void SaveToSettings()
    {
        _settings.UpscaleFactor = UpscaleFactor;
        _settings.UseRandomSeed = UseRandomSeed;
        _settings.Seed = long.TryParse(SeedText, out var seed) ? seed : null;
        _settings.OutputFormat = OutputFormatString switch
        {
            "jpg" => "Jpg",
            "webp" => "WebP",
            "heic" => "Heic",
            _ => "Png"
        };
        _settings.Save();
    }

    // ========== Lifecycle ==========
    public async void OnWindowLoaded()
    {
        try
        {
            await InitializeAsync();
        }
        catch (Exception ex)
        {
            Log($"初期化エラー: {ex.Message}", LogLevel.Error);
            InitializationFailed = true;
        }
    }

    public void OnWindowClosing()
    {
        SaveToSettings();
        _supirWrapper?.Dispose();
        Logger.Dispose();
    }

    // ========== Commands ==========
    private async Task InitializeAsync()
    {
        IsToolDownloading = true;
        ToolDownloadStatus = "SUPIR環境を確認中...";
        ToolDownloadProgress = null;

        try
        {
            // GPU検出 (パッケージインストールより先に実行)
            ToolDownloadStatus = "GPU を検出中...";
            var detectedDevice = GpuDetectionService.DetectOptimalDevice();

            // デバイス種別に応じたPython環境を構築
            _bootstrapService = new PythonBootstrapService(_settings, detectedDevice);
            var success = await Task.Run(() => _bootstrapService.InitializeAsync((step, total, msg, detail) =>
            {
                Dispatcher.UIThread.Post(() =>
                {
                    ToolDownloadStepText = $"ステップ {step}/{total}";
                    ToolDownloadStatus = msg;
                    ToolDownloadProgress = (double)step / total * 100;
                    if (detail != null)
                        ToolDownloadDetail = detail;
                });
            }));

            if (!success)
            {
                InitializationFailed = true;
                ToolDownloadStatus = "セットアップに失敗しました。";
                return;
            }

            // ワーカー起動 (検出済みデバイス種別を渡す)
            ToolDownloadStatus = "SUPIRワーカーを起動中...";
            ToolDownloadProgress = null;

            _supirWrapper = new SupirWrapperService(_bootstrapService.PythonHome, _bootstrapService, detectedDevice);
            await _supirWrapper.StartAsync();

            IsInitialized = true;
        }
        catch (Exception ex)
        {
            Log($"初期化失敗: {ex.Message}", LogLevel.Error);
            InitializationFailed = true;
            ToolDownloadStatus = $"エラー: {ex.Message}";
        }
        finally
        {
            IsToolDownloading = !IsInitialized && !InitializationFailed;
            if (IsInitialized)
                IsToolDownloading = false;
        }
    }

    [RelayCommand]
    private async Task RetryInitializeAsync()
    {
        InitializationFailed = false;
        await InitializeAsync();
    }

    // File picker commands - called from code-behind
    public void SetInputFilePath(string path)
    {
        InputFilePath = path;
    }

    public void SetInputFolderPath(string path)
    {
        InputFolderPath = path;
    }

    [RelayCommand(IncludeCancelCommand = true)]
    private async Task UpscaleImageAsync(CancellationToken ct)
    {
        if (_supirWrapper == null || !IsInitialized) return;

        SaveToSettings();
        IsProcessing = true;
        ProgressPercentage = null;

        _supirWrapper.OnProcessingProgress = (current, total) =>
        {
            Dispatcher.UIThread.Post(() =>
            {
                if (total > 0)
                    ProgressPercentage = (double)current / total * 100;
            });
        };

        try
        {
            long? seed = UseRandomSeed ? null :
                (long.TryParse(SeedText, out var s) ? s : null);

            if (IoFormMode == IoFormMode.FileSelection)
            {
                await ProcessSingleFile(InputFilePath, OutputFilePath, seed, ct);
            }
            else
            {
                await ProcessFolder(InputFolderPath, OutputFolderPath, seed, ct);
            }
        }
        catch (OperationCanceledException)
        {
            Log("処理がキャンセルされました。", LogLevel.Info);
        }
        catch (Exception ex)
        {
            Log($"処理エラー: {ex.Message}", LogLevel.Error);
        }
        finally
        {
            IsProcessing = false;
            ProgressPercentage = null;
            _supirWrapper.OnProcessingProgress = null;
        }
    }

    private async Task ProcessSingleFile(string input, string output, long? seed, CancellationToken ct)
    {
        if (string.IsNullOrEmpty(input) || !File.Exists(input))
        {
            Log("入力ファイルが見つかりません。", LogLevel.Warning);
            return;
        }

        var outputDir = Path.GetDirectoryName(output);
        if (!string.IsNullOrEmpty(outputDir))
            Directory.CreateDirectory(outputDir);

        Log($"処理中: {Path.GetFileName(input)}", LogLevel.Info);
        ProgressPercentage = null; // indeterminate

        await _supirWrapper!.ProcessImageAsync(
            input, output,
            UpscaleFactor, EdmSteps, GuidanceScale, seed,
            SStage1, SStage2, SChurn, SNoise,
            TiledSize, UseTiling, OutputFormatString, ct);

        ProgressPercentage = 100;
        Log($"完了: {output}", LogLevel.Info);
    }

    private async Task ProcessFolder(string inputFolder, string outputFolder, long? seed, CancellationToken ct)
    {
        if (string.IsNullOrEmpty(inputFolder) || !Directory.Exists(inputFolder))
        {
            Log("入力フォルダが見つかりません。", LogLevel.Warning);
            return;
        }

        Directory.CreateDirectory(outputFolder);

        var extensions = new[] { ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp" };
        var files = Directory.GetFiles(inputFolder)
            .Where(f => extensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .ToArray();

        if (files.Length == 0)
        {
            Log("画像ファイルが見つかりません。", LogLevel.Warning);
            return;
        }

        for (var i = 0; i < files.Length; i++)
        {
            ct.ThrowIfCancellationRequested();

            var file = files[i];
            var outputName = Path.GetFileNameWithoutExtension(file) + OutputExtension;
            var outputPath = Path.Combine(outputFolder, outputName);

            Log($"処理中 ({i + 1}/{files.Length}): {Path.GetFileName(file)}", LogLevel.Info);

            await _supirWrapper!.ProcessImageAsync(
                file, outputPath,
                UpscaleFactor, EdmSteps, GuidanceScale, seed,
                SStage1, SStage2, SChurn, SNoise,
                TiledSize, UseTiling, OutputFormatString, ct);

            ProgressPercentage = (double)(i + 1) / files.Length * 100;
        }

        Log($"一括処理完了: {files.Length}ファイル", LogLevel.Info);
    }

    // IoFormMode change handlers
    partial void OnIoFormModeChanged(IoFormMode value)
    {
        if (value == IoFormMode.FileSelection)
        {
            InputFolderPath = "";
            OutputFolderPath = "";
        }
        else
        {
            InputFilePath = "";
            OutputFilePath = "";
        }
    }
}
