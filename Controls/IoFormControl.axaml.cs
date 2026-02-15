using System.Globalization;
using Avalonia.Controls;
using Avalonia.Data.Converters;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using SUPIR_GUI.Models;
using SUPIR_GUI.ViewModels;

namespace SUPIR_GUI.Controls;

public partial class IoFormControl : UserControl
{
    public IoFormControl()
    {
        InitializeComponent();
    }

    private async void BrowseInputFile_Click(object? sender, RoutedEventArgs e)
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel == null) return;

        var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "画像ファイルを選択",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("画像ファイル")
                {
                    Patterns = new[] { "*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif", "*.bmp", "*.webp" }
                },
                FilePickerFileTypes.All
            }
        });

        if (files.Count > 0 && DataContext is MainWindowViewModel vm)
        {
            var path = files[0].TryGetLocalPath();
            if (!string.IsNullOrEmpty(path))
                vm.SetInputFilePath(path);
        }
    }

    private async void BrowseInputFolder_Click(object? sender, RoutedEventArgs e)
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel == null) return;

        var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "画像フォルダを選択",
            AllowMultiple = false
        });

        if (folders.Count > 0 && DataContext is MainWindowViewModel vm)
        {
            var path = folders[0].TryGetLocalPath();
            if (!string.IsNullOrEmpty(path))
                vm.SetInputFolderPath(path);
        }
    }
}

// IoFormMode <-> TabControl.SelectedIndex converter
public class IoFormModeToIndexConverter : IValueConverter
{
    public static readonly IoFormModeToIndexConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return value is IoFormMode mode ? (int)mode : 0;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return value is int index ? (IoFormMode)index : IoFormMode.FileSelection;
    }
}
