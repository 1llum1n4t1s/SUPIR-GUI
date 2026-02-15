using CommunityToolkit.Mvvm.ComponentModel;

namespace SUPIR_GUI.Models;

public partial class FileItem : ObservableObject
{
    [ObservableProperty]
    private string _path = "";

    [ObservableProperty]
    private string _status = "待機中";

    [ObservableProperty]
    private string _outputPath = "";

    public string Name => System.IO.Path.GetFileName(Path);

    public bool Exists() => File.Exists(Path);

    public bool IsImageFile()
    {
        var ext = System.IO.Path.GetExtension(Path).ToLowerInvariant();
        return ext is ".jpg" or ".jpeg" or ".png" or ".tiff" or ".tif" or ".bmp" or ".webp";
    }
}
