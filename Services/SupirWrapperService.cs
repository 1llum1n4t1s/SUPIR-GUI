using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using static SUPIR_GUI.Services.Logger;

namespace SUPIR_GUI.Services;

[SupportedOSPlatform("windows")]
public sealed partial class SupirWrapperService : IDisposable
{
    private readonly string _pythonHome;
    private readonly string _pythonExePath;
    private readonly string _workerScriptPath;
    private readonly PythonBootstrapService _bootstrapService;
    private Process? _workerProcess;
    private StreamWriter? _workerInput;
    private StreamReader? _workerOutput;
    private bool _initialized;
    private bool _disposed;
    private string? _currentDevice;
    private readonly SemaphoreSlim _processSemaphore = new(1, 1);

    public Action<int, int>? OnProcessingProgress { get; set; }

    [GeneratedRegex(@"(?:(?<percent>\d+)%\|.*?\|\s*)?(?<current>[\d\.]+)(?<unit>[kMG]B)?/(?<total>[\d\.]+)(?<total_unit>[kMG]B)?|(?<current_only>\d+)it \[")]
    private static partial Regex TqdmProgressRegex();

    /// <summary>
    /// コンストラクタ
    /// </summary>
    /// <param name="pythonHome">Pythonホームディレクトリ</param>
    /// <param name="bootstrapService">Pythonブートストラップサービス</param>
    /// <param name="detectedDevice">検出済みのGPUデバイス種別</param>
    public SupirWrapperService(string pythonHome, PythonBootstrapService bootstrapService, string detectedDevice)
    {
        _pythonHome = pythonHome;
        _pythonExePath = Path.Combine(_pythonHome, "python.exe");
        _bootstrapService = bootstrapService;
        _workerScriptPath = ResolveWorkerScriptPath();
        _currentDevice = detectedDevice;
    }

    /// <summary>
    /// ワーカープロセスを起動し、pingで接続を確認する
    /// </summary>
    public async Task StartAsync()
    {
        if (_initialized) return;

        Log("SUPIRワーカープロセスを起動中...", LogLevel.Info);

        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonExePath,
            Arguments = $"-u \"{_workerScriptPath}\"",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardInputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            StandardOutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            StandardErrorEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)
        };

        _bootstrapService.ConfigurePythonEnvironment(startInfo);

        _workerProcess = new Process { StartInfo = startInfo };
        _workerProcess.Start();

        Log($"ワーカープロセス起動 (PID: {_workerProcess.Id})", LogLevel.Debug);

        _workerInput = _workerProcess.StandardInput;
        _workerOutput = _workerProcess.StandardOutput;

        // stderrをバックグラウンドで収集（tqdm進捗パース + エラー診断用）
        // 診断用バッファは最大64KB に制限（メモリ無制限増大を防止）
        const int MaxStderrChars = 64 * 1024;
        var stderrCollector = new StringBuilder();
        _ = Task.Run(() => MonitorStderr(_workerProcess.StandardError, stderrCollector, MaxStderrChars));

        // プロセスが即座に終了していないか確認
        if (_workerProcess.HasExited)
        {
            var exitCode = _workerProcess.ExitCode;
            var stderrOutput = stderrCollector.ToString();
            Log($"ワーカープロセスが起動直後に終了 (終了コード: {exitCode})", LogLevel.Error);
            if (!string.IsNullOrWhiteSpace(stderrOutput))
                Log($"ワーカーstderr:\n{stderrOutput}", LogLevel.Error);
            throw new InvalidOperationException(
                $"ワーカープロセスが起動直後に終了しました (終了コード: {exitCode})。{(string.IsNullOrWhiteSpace(stderrOutput) ? "" : $"\n{stderrOutput.Trim()}")}");
        }

        // pingで接続確認
        var response = await SendCommandAsync(new Dictionary<string, object?>
        {
            ["command"] = "ping"
        }, TimeSpan.FromSeconds(120));

        if (response.Status != "ok")
        {
            var stderrOutput = stderrCollector.ToString();
            var details = !string.IsNullOrWhiteSpace(response.Message) ? response.Message : stderrOutput.Trim();
            throw new InvalidOperationException(
                $"ワーカーの起動に失敗しました: {(string.IsNullOrWhiteSpace(details) ? "応答なし" : details)}");
        }

        _initialized = true;
        Log("SUPIRワーカーの起動が完了しました。", LogLevel.Info);
    }

    public async Task ProcessImageAsync(
        string inputPath,
        string outputPath,
        int upscaleFactor,
        int edmSteps,
        double guidanceScale,
        long? seed,
        double sStage1,
        double sStage2,
        double sChurn,
        double sNoise,
        int tiledSize,
        bool useTiling,
        string outputFormat,
        CancellationToken ct = default)
    {
        await _processSemaphore.WaitAsync(ct);
        try
        {
            if (!_initialized)
                throw new InvalidOperationException("ワーカーが初期化されていません。");

            var payload = new Dictionary<string, object?>
            {
                ["command"] = "process",
                ["input"] = inputPath,
                ["output"] = outputPath,
                ["device"] = _currentDevice,
                ["upscale_factor"] = upscaleFactor,
                ["edm_steps"] = edmSteps,
                ["guidance_scale"] = guidanceScale,
                ["seed"] = seed,
                ["s_stage1"] = sStage1,
                ["s_stage2"] = sStage2,
                ["s_churn"] = sChurn,
                ["s_noise"] = sNoise,
                ["tiled_size"] = tiledSize,
                ["use_tiling"] = useTiling,
                ["output_format"] = outputFormat
            };

            var response = await SendCommandAsync(payload, TimeSpan.FromMinutes(30), ct);

            if (response.Status != "ok")
            {
                throw new InvalidOperationException(
                    $"画像処理に失敗しました: {response.Message}");
            }
        }
        finally
        {
            _processSemaphore.Release();
        }
    }

    /// <summary>
    /// ワーカーにコマンドを送信し、応答を待機する
    /// </summary>
    /// <param name="payload">送信するコマンドペイロード</param>
    /// <param name="timeout">タイムアウト時間</param>
    /// <param name="ct">キャンセルトークン</param>
    /// <returns>ワーカーからの応答</returns>
    private async Task<WorkerResponse> SendCommandAsync(
        Dictionary<string, object?> payload,
        TimeSpan? timeout = null,
        CancellationToken ct = default)
    {
        if (_workerInput == null || _workerOutput == null)
            throw new InvalidOperationException("ワーカープロセスが起動していません。");

        var json = JsonSerializer.Serialize(payload);
        await _workerInput.WriteLineAsync(json);
        await _workerInput.FlushAsync();

        var effectiveTimeout = timeout ?? TimeSpan.FromMinutes(5);
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(effectiveTimeout);

        try
        {
            while (!cts.Token.IsCancellationRequested)
            {
                var line = await _workerOutput.ReadLineAsync(cts.Token);
                if (line == null)
                {
                    // プロセスが終了した場合、終了コードを含めて報告
                    var exitInfo = _workerProcess?.HasExited == true
                        ? $" (終了コード: {_workerProcess.ExitCode})"
                        : "";
                    throw new InvalidOperationException(
                        $"ワーカープロセスが予期せず終了しました{exitInfo}。");
                }

                line = line.Trim();
                if (string.IsNullOrEmpty(line)) continue;

                try
                {
                    var response = JsonSerializer.Deserialize<WorkerResponse>(line);
                    if (response == null) continue;

                    // statusフィールドが空のJSONはワーカーの応答ではない(Python起動時のノイズ)
                    if (string.IsNullOrEmpty(response.Status))
                    {
                        Log($"ワーカーからのstatusなしJSONデータ (スキップ): {line}", LogLevel.Debug);
                        continue;
                    }

                    // progress メッセージ: ワーカーがまだ動作中であることを示す
                    // ハートビート。タイムアウトをリセットして待機を継続する。
                    if (response.Status == "progress")
                    {
                        Log($"ワーカー進捗: {response.Message}", LogLevel.Debug);
                        // CancelAfter を再設定してタイムアウトをリセット
                        cts.CancelAfter(effectiveTimeout);
                        continue;
                    }

                    return response;
                }
                catch (JsonException)
                {
                    Log($"ワーカーからの非JSONデータ (スキップ): {line}", LogLevel.Debug);
                }
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"ワーカーからの応答がタイムアウトしました ({effectiveTimeout.TotalSeconds}秒)。");
        }

        throw new OperationCanceledException("操作がキャンセルされました。");
    }

    /// <summary>
    /// stderrを監視し、tqdm進捗をパースしつつエラー診断用に収集する
    /// </summary>
    /// <param name="stderr">stderrのストリームリーダー</param>
    /// <param name="collector">stderr出力を収集するStringBuilder (診断用)</param>
    private void MonitorStderr(StreamReader stderr, StringBuilder? collector = null, int maxCollectorChars = int.MaxValue)
    {
        try
        {
            while (stderr.ReadLine() is { } line)
            {
                Log($"[Worker] {line}", LogLevel.Debug);
                if (collector != null && collector.Length < maxCollectorChars)
                    collector.AppendLine(line);

                var match = TqdmProgressRegex().Match(line);
                if (match.Success)
                {
                    if (match.Groups["percent"].Success &&
                        int.TryParse(match.Groups["percent"].Value, out var percent))
                    {
                        if (match.Groups["current"].Success && match.Groups["total"].Success)
                        {
                            // tqdm は "1.5MB/2.0GB" のように小数値を出力する場合がある。
                            // double.TryParse (InvariantCulture) でパースし int に変換 (B2-BUG-17)
                            if (double.TryParse(match.Groups["current"].Value,
                                    System.Globalization.NumberStyles.Float,
                                    System.Globalization.CultureInfo.InvariantCulture,
                                    out var currentD) &&
                                double.TryParse(match.Groups["total"].Value,
                                    System.Globalization.NumberStyles.Float,
                                    System.Globalization.CultureInfo.InvariantCulture,
                                    out var totalD))
                            {
                                OnProcessingProgress?.Invoke((int)currentD, (int)totalD);
                            }
                        }
                    }
                    else if (match.Groups["current_only"].Success &&
                             int.TryParse(match.Groups["current_only"].Value, out var currentOnly))
                    {
                        OnProcessingProgress?.Invoke(currentOnly, 0);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Log($"stderr監視エラー: {ex.Message}", LogLevel.Warning);
        }
    }

    private static string ResolveWorkerScriptPath()
    {
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var candidate = Path.Combine(baseDir, "python", "supir_worker.py");
        if (File.Exists(candidate)) return candidate;

        var fallback = Path.Combine(baseDir, "supir_worker.py");
        if (File.Exists(fallback)) return fallback;

        throw new FileNotFoundException("SUPIR ワーカースクリプトが見つかりません。", candidate);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        try
        {
            if (_workerProcess != null && !_workerProcess.HasExited)
            {
                // shutdown コマンドを送信
                try
                {
                    _workerInput?.WriteLine(JsonSerializer.Serialize(
                        new Dictionary<string, object?> { ["command"] = "shutdown" }));
                    _workerInput?.Flush();
                    _workerProcess.WaitForExit(3000);
                }
                catch { /* ignore */ }

                if (!_workerProcess.HasExited)
                {
                    _workerProcess.Kill();
                }
            }
        }
        catch { /* ignore */ }

        _workerInput?.Dispose();
        _workerOutput?.Dispose();
        _workerProcess?.Dispose();
        _processSemaphore.Dispose();

        Log("SUPIRワーカーを終了しました。", LogLevel.Info);
    }

    /// <summary>
    /// ワーカーからのJSON応答
    /// Pythonは小文字キーで送信するためJsonPropertyNameで明示的にマッピング
    /// </summary>
    private sealed class WorkerResponse
    {
        /// <summary>応答ステータス ("ok" または "error")</summary>
        [JsonPropertyName("status")]
        public string Status { get; set; } = "";

        /// <summary>応答メッセージ</summary>
        [JsonPropertyName("message")]
        public string Message { get; set; } = "";

        /// <summary>出力ファイルパス</summary>
        [JsonPropertyName("output")]
        public string? Output { get; set; }

        /// <summary>エラー時のトレースバック</summary>
        [JsonPropertyName("traceback")]
        public string? Traceback { get; set; }
    }
}
