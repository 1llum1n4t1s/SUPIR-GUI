using System.Globalization;
using NLog;
using NLog.Config;
using NLog.Targets;

namespace SUPIR_GUI.Services;

public enum LogLevel
{
    Debug,
    Info,
    Warning,
    Error
}

public static class Logger
{
    private static NLog.Logger? _logger;
    private static bool _isConfigured;
    private static string _appName = "SUPIR-GUI";

    private static readonly LogLevel MinLogLevel =
#if DEBUG
        LogLevel.Debug;
#else
        LogLevel.Warning;
#endif

    public static void Initialize(string? logDirectory = null, string filePrefix = "SUPIR-GUI")
    {
        if (_isConfigured) return;

        var effectiveLogDir = logDirectory ?? AppDomain.CurrentDomain.BaseDirectory;
        _appName = filePrefix;

        if (!Directory.Exists(effectiveLogDir))
            Directory.CreateDirectory(effectiveLogDir);

        var nlogConfig = new LoggingConfiguration();

        var fileTarget = new FileTarget("file")
        {
            FileName = Path.Combine(effectiveLogDir, $"{filePrefix}_${{date:format=yyyyMMdd}}.log"),
            ArchiveAboveSize = 10 * 1024 * 1024,
            ArchiveFileName = Path.Combine(effectiveLogDir, $"{filePrefix}_${{date:format=yyyyMMdd}}_{{##}}.log"),
            MaxArchiveFiles = 10,
            Layout = "${longdate} [${uppercase:${level}}] ${message}${onexception:inner=${newline}${exception:format=tostring}}",
            Encoding = System.Text.Encoding.UTF8
        };

        var consoleTarget = new ConsoleTarget("console")
        {
            Layout = "${longdate} [${uppercase:${level}}] ${message}${onexception:inner=${newline}${exception:format=tostring}}"
        };

        nlogConfig.AddTarget(fileTarget);
        nlogConfig.AddTarget(consoleTarget);
        nlogConfig.AddRule(NLog.LogLevel.Trace, NLog.LogLevel.Fatal, fileTarget);
        nlogConfig.AddRule(NLog.LogLevel.Trace, NLog.LogLevel.Fatal, consoleTarget);

        LogManager.Configuration = nlogConfig;
        _logger = LogManager.GetLogger(filePrefix);
        _isConfigured = true;

        Log("Logger initialized", LogLevel.Debug);

        CleanupOldLogFiles(effectiveLogDir, filePrefix, 7);
    }

    public static void Log(string message, LogLevel level = LogLevel.Info)
    {
        if (level < MinLogLevel) return;
        if (!_isConfigured) Initialize();
        _logger?.Log(ToNLogLevel(level), message);
    }

    public static void LogException(string message, Exception exception)
    {
        if (!_isConfigured) Initialize();
        _logger?.Error(exception, message);
    }

    public static void LogStartup(string[]? args = null)
    {
        if (LogLevel.Debug < MinLogLevel) return;
        if (!_isConfigured) Initialize();
        _logger?.Debug(
            $"""
            === {_appName} 起動ログ ===
            起動時刻: {DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}
            実行ファイルパス: {Environment.ProcessPath}
            引数数: {args?.Length ?? 0}
            """);
    }

    public static void Dispose()
    {
        LogManager.Shutdown();
        _isConfigured = false;
    }

    private static void CleanupOldLogFiles(string logDirectory, string filePrefix, int retentionDays)
    {
        if (retentionDays <= 0) return;
        try
        {
            var cutoffDate = DateTime.Now.Date.AddDays(-retentionDays);
            var logFiles = Directory.GetFiles(logDirectory, $"{filePrefix}_*.log");
            foreach (var file in logFiles)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(file);
                    var parts = fileName.Split('_');
                    if (parts.Length >= 2 && parts[1].Length == 8 &&
                        DateTime.TryParseExact(parts[1], "yyyyMMdd", null, DateTimeStyles.None, out var fileDate) &&
                        fileDate < cutoffDate)
                    {
                        File.Delete(file);
                    }
                }
                catch { /* skip */ }
            }
        }
        catch { /* skip */ }
    }

    private static NLog.LogLevel ToNLogLevel(LogLevel level) => level switch
    {
        LogLevel.Debug => NLog.LogLevel.Debug,
        LogLevel.Info => NLog.LogLevel.Info,
        LogLevel.Warning => NLog.LogLevel.Warn,
        LogLevel.Error => NLog.LogLevel.Error,
        _ => NLog.LogLevel.Info
    };
}
