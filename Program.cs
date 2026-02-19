using System;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using SUPIR_GUI.Services;
using Velopack;
using Velopack.Sources;

namespace SUPIR_GUI;

internal sealed class Program
{
    /// <summary>
    /// 更新チェックのタイムアウト時間（ミリ秒）
    /// </summary>
    private const int UpdateCheckTimeoutMs = 10000;

    [STAThread]
    public static int Main(string[] args)
    {
        VelopackApp.Build()
            .OnAfterInstallFastCallback(v => StartupRegistration.Register())
            .OnAfterUpdateFastCallback(v => StartupRegistration.Register())
            .OnBeforeUninstallFastCallback(v => StartupRegistration.Unregister())
            .Run();

        // サイレント更新チェックモード
        if (args.Length > 0 && args[0] == "--update-check")
        {
            RunSilentUpdateCheck();
            return 0;
        }

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return 0;
    }

    /// <summary>
    /// UI なしでサイレント更新チェックを実行する。
    /// Windows ログイン時のスタートアップから呼び出される。
    /// </summary>
    private static void RunSilentUpdateCheck()
    {
        try
        {
            Logger.Initialize();
            Logger.Log("サイレント更新チェックを開始します。", LogLevel.Info);

            var repoUrl = "https://github.com/1llum1n4t1s/SUPIR-GUI";
            var source = new GithubSource(repoUrl, string.Empty, false);
            var updateManager = new UpdateManager(source);

            if (!updateManager.IsInstalled)
            {
                Logger.Log("開発環境のため更新チェックをスキップしました。", LogLevel.Debug);
                return;
            }

            Logger.Log($"更新チェック: リポジトリ: {repoUrl}", LogLevel.Info);

            // 更新チェック（タイムアウト付き）
            UpdateInfo? updateInfo;
            try
            {
                var checkTask = updateManager.CheckForUpdatesAsync();
                var timeoutTask = Task.Delay(UpdateCheckTimeoutMs);
                var completed = Task.WhenAny(checkTask, timeoutTask).GetAwaiter().GetResult();
                if (completed == timeoutTask)
                {
                    Logger.Log("更新チェックがタイムアウトしました。", LogLevel.Warning);
                    return;
                }
                updateInfo = checkTask.GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                Logger.Log("更新チェックがタイムアウトしました。", LogLevel.Warning);
                return;
            }

            if (updateInfo == null)
            {
                Logger.Log("利用可能な更新はありません。", LogLevel.Info);
                return;
            }

            Logger.Log("新しいバージョンを検出しました。更新をダウンロードしています...", LogLevel.Info);

            // ダウンロード（10分タイムアウト）
            try
            {
                using var downloadCts = new CancellationTokenSource(TimeSpan.FromMinutes(10));
                updateManager.DownloadUpdatesAsync(updateInfo, null, downloadCts.Token).GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                Logger.Log("ダウンロードがタイムアウトしました。", LogLevel.Warning);
                return;
            }

            Logger.Log("ダウンロード完了。更新を適用します。", LogLevel.Info);
            updateManager.ApplyUpdatesAndExit(updateInfo);
        }
        catch (Exception ex)
        {
            Logger.LogException("サイレント更新チェック中にエラーが発生しました", ex);
        }
        finally
        {
            Logger.Dispose();
        }
    }

    // Avalonia configuration, don't remove; also used by visual designer.
    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
