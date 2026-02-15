using Avalonia.Controls;
using SUPIR_GUI.Services;
using SUPIR_GUI.ViewModels;

namespace SUPIR_GUI.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        Closing += OnClosing;
    }

    private void OnLoaded(object? sender, EventArgs e)
    {
        if (DataContext is MainWindowViewModel vm)
        {
            Logger.Initialize();
            Logger.LogStartup();
            vm.OnWindowLoaded();
        }
    }

    private void OnClosing(object? sender, WindowClosingEventArgs e)
    {
        if (DataContext is MainWindowViewModel vm)
        {
            vm.OnWindowClosing();
        }
    }
}
