using System.Globalization;
using Avalonia.Data.Converters;

namespace SUPIR_GUI.Converters;

public class NullableDoubleToIsIndeterminateConverter : IValueConverter
{
    public static readonly NullableDoubleToIsIndeterminateConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return value is null;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotSupportedException();
    }
}
