namespace VoiceIsolation.Numerics;

/// <summary>
/// Real FFT of a power-of-two size, computed in double precision (a radix-2 complex transform of the full size with
/// precomputed twiddles). Inputs and outputs are float32 like the PyTorch reference (<c>torch.stft</c> /
/// <c>torch.fft.rfft</c> / <c>irfft</c>); the double-precision arithmetic in between keeps the transform's own error well
/// below float rounding. Instances are immutable and thread-safe; per-call scratch is stack or thread-local.
/// </summary>
public sealed class RealFft
{
    private readonly double[] _cos;
    private readonly double[] _sin;
    private readonly int[] _reverse;

    [ThreadStatic] private static double[]? t_re;
    [ThreadStatic] private static double[]? t_im;

    public RealFft(int size)
    {
        if (size < 2 || (size & (size - 1)) != 0) throw new ArgumentOutOfRangeException(nameof(size), "Size must be a power of two.");
        Size = size;
        _cos = new double[size / 2];
        _sin = new double[size / 2];
        for (int k = 0; k < size / 2; k++)
        {
            double angle = -2.0 * Math.PI * k / size;
            _cos[k] = Math.Cos(angle);
            _sin[k] = Math.Sin(angle);
        }
        _reverse = new int[size];
        int bits = System.Numerics.BitOperations.Log2((uint)size);
        for (int i = 0; i < size; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
            _reverse[i] = r;
        }
    }

    /// <summary>The transform size <c>n</c>; a spectrum has <c>n / 2 + 1</c> bins.</summary>
    public int Size { get; }

    public int Bins => Size / 2 + 1;

    /// <summary>Spectrum of <paramref name="input"/> (<see cref="Size"/> samples) as interleaved (re, im) pairs,
    /// <see cref="Bins"/> of them.</summary>
    public void Forward(ReadOnlySpan<float> input, Span<float> spectrum)
    {
        if (input.Length != Size || spectrum.Length < 2 * Bins) throw new ArgumentException("Bad buffer sizes.");
        var (re, im) = Scratch();
        for (int i = 0; i < Size; i++)
        {
            re[_reverse[i]] = input[i];
            im[_reverse[i]] = 0;
        }
        Transform(re, im, inverse: false);
        for (int k = 0; k < Bins; k++)
        {
            spectrum[2 * k] = (float)re[k];
            spectrum[2 * k + 1] = (float)im[k];
        }
    }

    /// <summary>
    /// Inverse of a one-sided spectrum (<see cref="Bins"/> interleaved pairs) into <see cref="Size"/> real samples, scaled by
    /// <c>1/n</c>. Like <c>torch.fft.irfft</c>, the imaginary parts of the DC and Nyquist bins are ignored.
    /// </summary>
    public void Inverse(ReadOnlySpan<float> spectrum, Span<float> output)
    {
        if (output.Length != Size || spectrum.Length < 2 * Bins) throw new ArgumentException("Bad buffer sizes.");
        var (re, im) = Scratch();
        int half = Size / 2;
        for (int k = 0; k <= half; k++)
        {
            double r = spectrum[2 * k], m = k == 0 || k == half ? 0 : spectrum[2 * k + 1];
            re[_reverse[k]] = r;
            im[_reverse[k]] = m;
            if (k > 0 && k < half)
            {
                re[_reverse[Size - k]] = r;
                im[_reverse[Size - k]] = -m;
            }
        }
        Transform(re, im, inverse: true);
        double scale = 1.0 / Size;
        for (int i = 0; i < Size; i++) output[i] = (float)(re[i] * scale);
    }

    private (double[] Re, double[] Im) Scratch()
    {
        if (t_re == null || t_re.Length < Size)
        {
            t_re = new double[Size];
            t_im = new double[Size];
        }
        return (t_re, t_im!);
    }

    private void Transform(double[] re, double[] im, bool inverse)
    {
        int n = Size;
        double sign = inverse ? -1 : 1;
        for (int len = 2; len <= n; len <<= 1)
        {
            int halfLen = len >> 1, stride = n / len;
            for (int i = 0; i < n; i += len)
            {
                for (int j = 0; j < halfLen; j++)
                {
                    double wr = _cos[j * stride], wi = sign * _sin[j * stride];
                    int a = i + j, b = a + halfLen;
                    double xr = re[b] * wr - im[b] * wi;
                    double xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
    }
}
