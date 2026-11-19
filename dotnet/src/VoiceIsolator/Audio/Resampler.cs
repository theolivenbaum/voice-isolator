namespace VoiceIsolation.Audio;

/// <summary>
/// Band-limited resampling with torchaudio's <c>kaiser_best</c> settings, the resampler of the Python package
/// (<c>audio.resample</c>): a Kaiser-windowed sinc kernel per output phase (<c>lowpass_filter_width = 64</c>,
/// <c>rolloff = 0.9475937</c>, <c>beta = 14.7696565</c>), applied as <c>torchaudio.functional.resample</c> does
/// (zero padding of <c>width</c> before and <c>width + orig</c> after, stride <c>orig</c>, <c>ceil(new · n / orig)</c>
/// outputs). The kernel is computed in double precision and applied in float32.
/// </summary>
public static class Resampler
{
    private const int LowpassFilterWidth = 64;
    private const double Rolloff = 0.9475937167399596;
    private const double Beta = 14.769656459379492;

    /// <summary>Resamples mono audio; returns the input itself when the rates match.</summary>
    public static float[] Resample(float[] input, int fromRate, int toRate)
    {
        if (fromRate <= 0 || toRate <= 0) throw new ArgumentOutOfRangeException(nameof(fromRate), "Sample rates must be positive.");
        if (fromRate == toRate) return input;
        int gcd = Gcd(fromRate, toRate);
        int orig = fromRate / gcd, next = toRate / gcd;
        var (kernel, width) = Kernel(orig, next);
        int taps = 2 * width + orig;
        int length = input.Length;
        long target = (long)Math.Ceiling((double)next * length / orig);
        var padded = new float[length + 2 * width + orig];
        input.CopyTo(padded, width);
        int positions = (padded.Length - taps) / orig + 1;
        var output = new float[target];
        Parallel.For(0, (positions + 255) / 256, chunk =>
        {
            int i0 = chunk * 256, i1 = Math.Min(positions, i0 + 256);
            for (int i = i0; i < i1; i++)
            {
                var window = padded.AsSpan(i * orig, taps);
                for (int j = 0; j < next; j++)
                {
                    long at = (long)i * next + j;
                    if (at >= target) break;
                    output[at] = System.Numerics.Tensors.TensorPrimitives.Dot(window, kernel.AsSpan(j * taps, taps));
                }
            }
        });
        return output;
    }

    /// <summary>The kernel <c>[new, 2·width + orig]</c> of <c>_get_sinc_resample_kernel</c> (sinc_interp_kaiser).</summary>
    internal static (float[] Kernel, int Width) Kernel(int orig, int next)
    {
        double baseFreq = Math.Min(orig, next) * Rolloff;
        int width = (int)Math.Ceiling(LowpassFilterWidth * orig / baseFreq);
        int taps = 2 * width + orig;
        var kernel = new float[next * taps];
        double i0Beta = BesselI0(Beta);
        double scale = baseFreq / orig;
        for (int j = 0; j < next; j++)
        {
            for (int m = 0; m < taps; m++)
            {
                double idx = (double)(m - width) / orig;
                double t = (-(double)j / next + idx) * baseFreq;
                t = Math.Clamp(t, -LowpassFilterWidth, LowpassFilterWidth);
                double ratio = t / LowpassFilterWidth;
                double window = BesselI0(Beta * Math.Sqrt(1 - ratio * ratio)) / i0Beta;
                double x = t * Math.PI;
                double sinc = x == 0 ? 1.0 : Math.Sin(x) / x;
                kernel[j * taps + m] = (float)(sinc * window * scale);
            }
        }
        return (kernel, width);
    }

    /// <summary>The modified Bessel function of the first kind, order zero (power series).</summary>
    internal static double BesselI0(double x)
    {
        double sum = 1, term = 1, q = x * x / 4;
        for (int k = 1; k < 500; k++)
        {
            term *= q / ((double)k * k);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }
}
