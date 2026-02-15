using System.Text.Json;

namespace SUPIR_GUI.Models;

public class AppSettings
{
    // Python runtime
    public string PythonHome { get; set; } = "";
    public string PythonVersion { get; set; } = "";

    // SUPIR parameters
    public int UpscaleFactor { get; set; } = 2;
    public long? Seed { get; set; }
    public bool UseRandomSeed { get; set; } = true;

    // Output settings
    public string OutputFormat { get; set; } = "Png";
    public string OutputFolder { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Pictures");
    public bool OverwriteOutputFiles { get; set; }
    public string OutputSuffix { get; set; } = "_supir";

    private static readonly string SettingsFilePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "SUPIR-GUI",
        "settings.json");

    public static AppSettings Load()
    {
        try
        {
            var settingsDir = Path.GetDirectoryName(SettingsFilePath) ?? "";
            if (!Directory.Exists(settingsDir))
                Directory.CreateDirectory(settingsDir);

            if (File.Exists(SettingsFilePath))
            {
                var json = File.ReadAllText(SettingsFilePath);
                return JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"設定の読み込み中にエラーが発生しました: {ex.Message}");
        }

        return new AppSettings();
    }

    public bool Save()
    {
        try
        {
            var settingsDir = Path.GetDirectoryName(SettingsFilePath) ?? "";
            if (!Directory.Exists(settingsDir))
                Directory.CreateDirectory(settingsDir);

            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(this, options);
            File.WriteAllText(SettingsFilePath, json);
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"設定の保存中にエラーが発生しました: {ex.Message}");
            return false;
        }
    }
}
