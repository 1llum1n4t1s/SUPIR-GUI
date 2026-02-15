using System.Management;
using System.Runtime.Versioning;
using static SUPIR_GUI.Services.Logger;

namespace SUPIR_GUI.Services;

/// <summary>
/// GPU検出サービス - WMIでGPUを識別し最適なデバイスタイプを判定する
/// </summary>
[SupportedOSPlatform("windows")]
public static class GpuDetectionService
{
    /// <summary>NVIDIA GPU (CUDA) デバイス識別子</summary>
    public const string DeviceCuda = "cuda";

    /// <summary>DirectML デバイス識別子 (AMD/Intel GPU)</summary>
    public const string DeviceDirectMl = "dml";

    /// <summary>CPU デバイス識別子 (GPU未検出時のフォールバック)</summary>
    public const string DeviceCpu = "cpu";

    /// <summary>
    /// WMIでGPUを検出し、最適なデバイスタイプを返す
    /// </summary>
    /// <returns>デバイス識別子 ("cuda", "dml", "cpu")</returns>
    public static string DetectOptimalDevice()
    {
        try
        {
            Log("GPU検出を開始します...", LogLevel.Debug);

            using var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");
            List<string> gpuNames = [];
            foreach (var obj in searcher.Get())
            {
                var name = obj["Name"]?.ToString() ?? "";
                if (!string.IsNullOrEmpty(name))
                {
                    gpuNames.Add(name.ToLower());
                    Log($"検出されたGPU: {name}", LogLevel.Debug);
                }
                obj?.Dispose();
            }

            // NVIDIA GPU → CUDA
            if (gpuNames.Any(g => g.Contains("geforce") || g.Contains("nvidia")))
            {
                Log("NVIDIA GPU検出。CUDA を使用します。", LogLevel.Info);
                return DeviceCuda;
            }

            // AMD/Intel GPU → DirectML
            if (gpuNames.Any(g => g.Contains("radeon") || g.Contains("amd") || g.Contains("intel")))
            {
                Log("AMD/Intel GPU検出。DirectML を使用します。", LogLevel.Info);
                return DeviceDirectMl;
            }
        }
        catch (Exception ex)
        {
            Log($"GPU検出中にエラーが発生しました: {ex.Message}", LogLevel.Warning);
        }

        Log("GPU検出失敗。CPU を使用します。", LogLevel.Info);
        return DeviceCpu;
    }
}
