using VoiceIsolation.IO;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>A linear layer <c>y = x·Wᵀ + b</c>: the PyTorch weight <c>W[out, in]</c> packed once for the GEMM.</summary>
internal sealed unsafe class Linear : IDisposable
{
    private readonly NativeBuffer<float>? _bias;

    public Linear(float* weight, int outputs, int inputs, float* bias)
    {
        In = inputs;
        Out = outputs;
        Weight = PackedMatrix.FromLinear(weight, outputs, inputs);
        if (bias != null)
        {
            _bias = new NativeBuffer<float>(outputs, zero: false, hugePages: false);
            new ReadOnlySpan<float>(bias, outputs).CopyTo(_bias.AsSpan());
        }
    }

    public int In { get; }
    public int Out { get; }
    public PackedMatrix Weight { get; }
    public float* Bias => _bias == null ? null : _bias.Ptr;

    /// <summary>Loads <c>{prefix}.weight</c> (and <c>{prefix}.bias</c> when <paramref name="bias"/>).</summary>
    public static Linear Load(SafeTensorsFile file, string prefix, int outputs, int inputs, bool bias = true) =>
        new(file.Float(prefix + ".weight", outputs, inputs), outputs, inputs, bias ? file.Float(prefix + ".bias", outputs) : null);

    /// <summary><c>y = act(x·Wᵀ + b)</c>, or <c>y = act(y + x·Wᵀ + b)</c> when <paramref name="accumulate"/>.</summary>
    public void Forward(float* x, long ldx, int rows, float* y, long ldy, Parallelism par, Activation act = Activation.None,
                        bool accumulate = false, float* biasOverride = null)
        => Gemm.Multiply(x, ldx, rows, Weight, y, ldy, par, biasOverride != null ? biasOverride : Bias, accumulate, act);

    public void Dispose()
    {
        Weight.Dispose();
        _bias?.Dispose();
    }
}

/// <summary>A LayerNorm's affine parameters (<c>eps = 1e-5</c>, PyTorch's default).</summary>
internal sealed unsafe class Norm : IDisposable
{
    public const float Eps = 1e-5f;
    private readonly NativeBuffer<float> _params;

    public Norm(float* gamma, float* beta, int dim)
    {
        Dim = dim;
        _params = new NativeBuffer<float>(2L * dim, zero: false, hugePages: false);
        new ReadOnlySpan<float>(gamma, dim).CopyTo(_params.AsSpan(0, dim));
        new ReadOnlySpan<float>(beta, dim).CopyTo(_params.AsSpan(dim, dim));
    }

    public int Dim { get; }
    public float* Gamma => _params.Ptr;
    public float* Beta => _params.Ptr + Dim;

    public static Norm Load(SafeTensorsFile file, string prefix, int dim) =>
        new(file.Float(prefix + ".weight", dim), file.Float(prefix + ".bias", dim), dim);

    public void Forward(float* x, float* y, int rows, Parallelism par) => Ops.LayerNorm(x, y, rows, Dim, Gamma, Beta, Eps, par);

    public void Forward(float* x, long ldx, float* y, long ldy, int rows)
    {
        for (int r = 0; r < rows; r++) Ops.LayerNormRow(x + r * ldx, y + r * ldy, Dim, Gamma, Beta, Eps);
    }

    public void Dispose() => _params.Dispose();
}

/// <summary>A float vector owned in aligned native memory (biases, embeddings, Mamba constants).</summary>
internal sealed unsafe class Vec : IDisposable
{
    private readonly NativeBuffer<float> _data;

    public Vec(ReadOnlySpan<float> values)
    {
        _data = new NativeBuffer<float>(values.Length, zero: false, hugePages: false);
        values.CopyTo(_data.AsSpan());
    }

    public Vec(int length) => _data = new NativeBuffer<float>(length, zero: true, hugePages: false);

    public int Length => (int)_data.Length;
    public float* Ptr => _data.Ptr;
    public Span<float> Span => _data.AsSpan();

    public void Dispose() => _data.Dispose();
}
