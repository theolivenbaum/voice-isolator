using System.Numerics.Tensors;
using VoiceIsolation.Audio;
using VoiceIsolation.IO;
using VoiceIsolation.Model;
using VoiceIsolation.Numerics;
using Xunit.Abstractions;

namespace VoiceIsolation.Tests;

/// <summary>Kernel tests against naive double-precision references; they need no model and always run.</summary>
public sealed unsafe class KernelTests(ITestOutputHelper output)
{
    private static float[] Random(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return x;
    }

    [Theory]
    [InlineData(1, 64, 32, Activation.None)]
    [InlineData(5, 512, 1536, Activation.None)]
    [InlineData(37, 300, 100, Activation.Gelu)]
    [InlineData(301, 512, 2048, Activation.Gelu)]
    [InlineData(13, 1024, 64, Activation.Silu)]
    public void GemmMatchesNaive(int m, int k, int n, Activation act)
    {
        float[] a = Random(m * k, 1), w = Random(n * k, 2, 0.1f), bias = Random(n, 3), c = new float[m * n];
        fixed (float* ap = a, wp = w, bp = bias, cp = c)
        {
            using var packed = PackedMatrix.FromLinear(wp, n, k);
            Gemm.Multiply(ap, k, m, packed, cp, n, Parallelism.Single, bp, act: act);
        }
        var expected = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = bias[j];
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * w[j * k + p];
                expected[i * n + j] = act switch
                {
                    Activation.Gelu => (float)(s * 0.5 * (1 + ErfReference(s / Math.Sqrt(2)))),
                    Activation.Silu => (float)(s / (1 + Math.Exp(-s))),
                    _ => (float)s,
                };
            }
        Fixtures.AssertClose(output, $"gemm {m}x{k}x{n} {act}", expected, c, 1e-5);
    }

    private static double ErfReference(double x)
    {
        // Abramowitz & Stegun 7.1.28 is too coarse; integrate the series instead (|x| is small in these tests).
        double sum = 0, term = x;
        for (int n = 0; n < 200; n++)
        {
            sum += term / (2 * n + 1);
            term *= -x * x / (n + 1);
        }
        return 2 / Math.Sqrt(Math.PI) * sum;
    }

    /// <summary>log(1 + e), accurate for small e (where 1 + e rounds).</summary>
    private static double Log1p(double e) => e < 1e-4 ? e - e * e / 2 + e * e * e / 3 : Math.Log(1 + e);

    [Fact]
    public void ErfAndGeluAreAccurate()
    {
        double worstErf = 0, worstSoftplus = 0;
        for (int i = -4000; i <= 4000; i++)
        {
            float x = i / 1000f;
            double reference = ErfReference(x);
            worstErf = Math.Max(worstErf, Math.Abs(VMath.Erf(x) - reference));
            if (Vector512.IsHardwareAccelerated)
            {
                worstErf = Math.Max(worstErf, Math.Abs(VMath.Erf(Vector512.Create(x)).GetElement(0) - reference));
                float xs = x * 10f;
                float sp = VMath.Softplus(Vector512.Create(xs)).GetElement(0);
                double e = Math.Exp((double)xs), spRef = Log1p(e);
                worstSoftplus = Math.Max(worstSoftplus, Math.Abs(sp - spRef) / Math.Max(1e-30, Math.Abs(spRef)));
            }
        }
        output.WriteLine($"erf max abs error {worstErf:E2}, softplus max rel error {worstSoftplus:E2}");
        Assert.True(worstErf < 5e-7);   // A&S 7.1.26 (1.5e-7) plus float rounding, as PyTorch's CPU erf
        Assert.True(worstSoftplus < 1e-6);
    }

    [Fact]
    public void RealFftMatchesDft()
    {
        const int n = 64;
        var fft = new RealFft(n);
        float[] x = Random(n, 4);
        var spectrum = new float[2 * (n / 2 + 1)];
        fft.Forward(x, spectrum);
        for (int k = 0; k <= n / 2; k++)
        {
            double re = 0, im = 0;
            for (int t = 0; t < n; t++)
            {
                re += x[t] * Math.Cos(-2 * Math.PI * k * t / n);
                im += x[t] * Math.Sin(-2 * Math.PI * k * t / n);
            }
            Assert.Equal(re, spectrum[2 * k], 4);
            Assert.Equal(im, spectrum[2 * k + 1], 4);
        }
        var back = new float[n];
        fft.Inverse(spectrum, back);
        Fixtures.AssertClose(output, "irfft(rfft(x))", x, back, 1e-6);
    }

    [Fact]
    public void ModelArchiveRoundTrips()
    {
        // a tiny safetensors file: header, then 1001 floats (a tail of 4 bytes stays unshuffled)
        string dir = Path.Combine(Path.GetTempPath(), "vi-archive-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            float[] values = Random(1001, 5);
            string header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[1001],\"data_offsets\":[0,4004]},\"__metadata__\":{\"config\":\"{}\"}}";
            var bytes = new List<byte>(BitConverter.GetBytes((long)header.Length));
            bytes.AddRange(System.Text.Encoding.UTF8.GetBytes(header));
            bytes.AddRange(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
            string path = Path.Combine(dir, "m.safetensors");
            File.WriteAllBytes(path, bytes.ToArray());
            using var packed = new MemoryStream();
            ModelArchive.Pack(path, packed);
            packed.Position = 0;
            Assert.True(ModelArchive.IsArchive(packed.GetBuffer()));
            using var file = SafeTensorsFile.FromBuffer(ModelArchive.Unpack(packed), "m");
            Assert.Equal(values, file.FloatArray("x", 1001));
            Assert.Equal("{}", file.Metadata["config"]);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void ResamplerKeepsATone()
    {
        // 1 kHz at 16 kHz, up to 48 kHz and back: the tone survives, the length is ceil(n · new / orig)
        var x = new float[16000];
        for (int i = 0; i < x.Length; i++) x[i] = (float)Math.Sin(2 * Math.PI * 1000 * i / 16000.0);
        float[] up = Resampler.Resample(x, 16000, 48000);
        Assert.Equal(48000, up.Length);
        float[] down = Resampler.Resample(up, 48000, 16000);
        Assert.Equal(16000, down.Length);
        Fixtures.AssertClose(output, "16k→48k→16k (middle)", x.AsSpan(1000, 14000), down.AsSpan(1000, 14000), 1e-3);
        float[] odd = Resampler.Resample(x, 16000, 44100);
        Assert.Equal((int)Math.Ceiling(16000 * 44100 / 16000.0), odd.Length);
    }

    [Fact]
    public void LinspaceMatchesTorch()
    {
        var w = new float[5];
        VoiceIsolatorNetwork.Linspace(w, 0f, 1f);
        Assert.Equal([0f, 0.25f, 0.5f, 0.75f, 1f], w);
    }

    [Fact]
    public void BandEdgesMatchTheReleasedLayout()
    {
        var config = new VoiceIsolatorConfig { Width = 512 };
        var edges = config.BandEdges();
        Assert.Equal(46, edges.Length);
        Assert.Equal((0, 4), edges[0]);
        Assert.Equal(1025, edges[^1].End);
        for (int i = 1; i < edges.Length; i++) Assert.Equal(edges[i - 1].End, edges[i].Start);
    }

    [Fact]
    public void SoftplusOfSpanMatchesScalar()
    {
        float[] x = Random(1000, 6, 30f);
        float[] y = (float[])x.Clone();
        MambaLayer.Softplus(y);
        float[] expected = x.Select(v => v > 20 ? v : (float)Log1p(Math.Exp(v))).ToArray();
        Fixtures.AssertClose(output, "softplus", expected, y, 1e-6);
        Assert.True(TensorPrimitives.Min<float>(y) >= 0);
    }
}
