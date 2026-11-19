using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>
/// The STFT front end and iSTFT back end (<c>torch.stft</c> / <c>torch.istft</c> with <c>center=True</c> and a periodic
/// Hann window), and the correction the decoder's output encodes (<c>VoiceIsolatorNetwork.correction</c>).
/// Spectra are <c>[frames, bins, 2]</c> (frame-major, interleaved real and imaginary parts).
/// </summary>
internal sealed unsafe class Spectral
{
    public Spectral(int nFft, int hop, float correctionBound)
    {
        NFft = nFft;
        Hop = hop;
        Bound = correctionBound;
        Fft = new RealFft(nFft);
        Window = new float[nFft];
        for (int i = 0; i < nFft; i++) Window[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / nFft));
        WindowSquared = Window.Select(w => w * w).ToArray();
    }

    public int NFft { get; }
    public int Hop { get; }
    public int Bins => NFft / 2 + 1;
    public float Bound { get; }
    public RealFft Fft { get; }
    public float[] Window { get; }
    public float[] WindowSquared { get; }

    /// <summary>Frames of a clip of <paramref name="samples"/> samples: <c>1 + samples / hop</c>.</summary>
    public int Frames(int samples) => 1 + samples / Hop;

    /// <summary>
    /// <c>torch.stft(center=True)</c>: frame <c>t</c> windows samples <c>[t·hop - n/2, t·hop + n/2)</c> of the clip padded by
    /// reflection (by zeros when the clip is no longer than <c>n/2</c>, as the reference does).
    /// </summary>
    public void Stft(ReadOnlySpan<float> audio, float* spectrum, Parallelism par)
    {
        int n = NFft, half = n / 2, length = audio.Length, frames = Frames(length);
        bool reflect = length > half;
        float[] padded = new float[length + n];
        audio.CopyTo(padded.AsSpan(half));
        for (int i = 0; i < half; i++)
        {
            padded[half - 1 - i] = reflect ? audio[i + 1] : 0f;
            padded[half + length + i] = reflect ? audio[length - 2 - i] : 0f;
        }
        long ld = 2L * Bins;
        par.ForRange(frames, 8, (f0, f1) =>
        {
            Span<float> segment = stackalloc float[n];
            for (int t = f0; t < f1; t++)
            {
                System.Numerics.Tensors.TensorPrimitives.Multiply(padded.AsSpan(t * Hop, n), Window, segment);
                Fft.Forward(segment, new Span<float>(spectrum + t * ld, (int)ld));
            }
        });
    }

    /// <summary>Spectrum of one windowed frame of <paramref name="segment"/> (<c>n</c> samples, zeros where there are none).</summary>
    public void AnalyzeFrame(ReadOnlySpan<float> segment, Span<float> spectrum)
    {
        Span<float> windowed = stackalloc float[NFft];
        System.Numerics.Tensors.TensorPrimitives.Multiply(segment, Window, windowed);
        Fft.Forward(windowed, spectrum);
    }

    /// <summary>
    /// The correction <c>spectrum · (2 tanh r₀ + 2i tanh r₁) + residual</c>, where the residual
    /// <c>(tanh r₂, tanh r₃) / max(1, ‖·‖) · bound · |spectrum|</c> is bounded relative to its bin, for
    /// <paramref name="frames"/> frames of <paramref name="raw"/> <c>[frames, bins, 4]</c>.
    /// </summary>
    public void Correction(float* raw, long ldRaw, float* spectrum, long ldSpectrum, int frames, float* output, long ldOutput,
                           Parallelism par)
    {
        int bins = Bins;
        float bound = Bound;
        par.ForRange(frames, 16, (f0, f1) =>
        {
            Span<float> t = stackalloc float[4 * bins];
            for (int f = f0; f < f1; f++)
            {
                System.Numerics.Tensors.TensorPrimitives.Tanh(new ReadOnlySpan<float>(raw + f * ldRaw, 4 * bins), t);
                float* s = spectrum + f * ldSpectrum, o = output + f * ldOutput;
                for (int k = 0; k < bins; k++)
                {
                    float re = s[2 * k], im = s[2 * k + 1];
                    float mr = 2f * t[4 * k], mi = 2f * t[4 * k + 1];
                    float cr = t[4 * k + 2], ci = t[4 * k + 3];
                    float norm = MathF.Max(MathF.Sqrt(cr * cr + ci * ci), 1f);
                    float magnitude = (float)Math.Sqrt((double)re * re + (double)im * im);
                    float scale = bound * magnitude;
                    o[2 * k] = (re * mr - im * mi) + cr / norm * scale;
                    o[2 * k + 1] = (re * mi + im * mr) + ci / norm * scale;
                }
            }
        });
    }

    /// <summary>One frame's inverse transform times the window (<c>irfft(spectrum) · window</c>).</summary>
    public void SynthesizeFrame(ReadOnlySpan<float> spectrum, Span<float> frame)
    {
        Fft.Inverse(spectrum, frame);
        System.Numerics.Tensors.TensorPrimitives.Multiply(frame, Window, frame);
    }

    /// <summary><c>torch.istft(center=True, length)</c> of <paramref name="frames"/> spectra: overlap-add of the windowed
    /// inverse transforms, divided by the summed squared window, trimmed by <c>n/2</c> at the start.</summary>
    public float[] Istft(float* spectrum, long ldSpectrum, int frames, int length, Parallelism par)
    {
        int n = NFft, hop = Hop;
        int total = n + hop * (frames - 1);
        var synthesized = new float[(long)frames * n];
        par.ForRange(frames, 8, (f0, f1) =>
        {
            for (int t = f0; t < f1; t++)
                SynthesizeFrame(new ReadOnlySpan<float>(spectrum + t * ldSpectrum, 2 * Bins), synthesized.AsSpan(t * n, n));
        });
        var ola = new float[total];
        var envelope = new float[total];
        for (int t = 0; t < frames; t++)
        {
            var dst = ola.AsSpan(t * hop, n);
            System.Numerics.Tensors.TensorPrimitives.Add(dst, synthesized.AsSpan(t * n, n), dst);
            var env = envelope.AsSpan(t * hop, n);
            System.Numerics.Tensors.TensorPrimitives.Add(env, WindowSquared, env);
        }
        var output = new float[length];
        int offset = n / 2;
        for (int i = 0; i < length; i++)
        {
            int at = i + offset;
            if (at >= total) break;
            output[i] = ola[at] / envelope[at];
        }
        return output;
    }
}
