using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace VoiceIsolation.Numerics;

/// <summary>Row-wise and element-wise kernels over channels-last activations <c>x[t, c]</c> (LayerNorm, softmax and
/// transpose adapted from curiosity-ai/parakeet, Apache-2.0).</summary>
public static unsafe class Ops
{
    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseAvx2 = Avx2.IsSupported && Fma.IsSupported;

    /// <summary><c>y[t] = (x[t] - mean) / sqrt(var + eps) · gamma + beta</c> over rows of length <paramref name="dim"/>.</summary>
    public static void LayerNorm(float* x, float* y, int rows, int dim, float* gamma, float* beta, float eps, Parallelism par)
    {
        if ((long)rows * dim < 1 << 14)
        {
            for (int r = 0; r < rows; r++) LayerNormRow(x + (long)r * dim, y + (long)r * dim, dim, gamma, beta, eps);
            return;
        }
        par.ForRange(rows, 16, (r0, r1) =>
        {
            for (int r = r0; r < r1; r++) LayerNormRow(x + (long)r * dim, y + (long)r * dim, dim, gamma, beta, eps);
        });
    }

    internal static void LayerNormRow(float* x, float* y, int dim, float* gamma, float* beta, float eps)
    {
        int i = 0;
        float mean, var;
        if (UseAvx512 && dim % 16 == 0)
        {
            var s0 = Vector512<float>.Zero; var s1 = Vector512<float>.Zero;
            for (; i + 32 <= dim; i += 32) { s0 += Vector512.Load(x + i); s1 += Vector512.Load(x + i + 16); }
            for (; i < dim; i += 16) s0 += Vector512.Load(x + i);
            mean = Vector512.Sum(s0 + s1) / dim;
            var m = Vector512.Create(mean);
            var q0 = Vector512<float>.Zero; var q1 = Vector512<float>.Zero;
            for (i = 0; i + 32 <= dim; i += 32)
            {
                var d0 = Vector512.Load(x + i) - m; var d1 = Vector512.Load(x + i + 16) - m;
                q0 = Avx512F.FusedMultiplyAdd(d0, d0, q0); q1 = Avx512F.FusedMultiplyAdd(d1, d1, q1);
            }
            for (; i < dim; i += 16) { var d0 = Vector512.Load(x + i) - m; q0 = Avx512F.FusedMultiplyAdd(d0, d0, q0); }
            var = Vector512.Sum(q0 + q1) / dim;
            var inv = Vector512.Create(1f / MathF.Sqrt(var + eps));
            for (i = 0; i < dim; i += 16)
            {
                var v = (Vector512.Load(x + i) - m) * inv;
                Vector512.Store(Avx512F.FusedMultiplyAdd(v, Vector512.Load(gamma + i), Vector512.Load(beta + i)), y + i);
            }
            return;
        }
        double s = 0;
        for (i = 0; i < dim; i++) s += x[i];
        mean = (float)(s / dim);
        double q = 0;
        for (i = 0; i < dim; i++) { float d = x[i] - mean; q += d * d; }
        var = (float)(q / dim);
        float invs = 1f / MathF.Sqrt(var + eps);
        for (i = 0; i < dim; i++) y[i] = (x[i] - mean) * invs * gamma[i] + beta[i];
    }

    /// <summary>In-place softmax of <paramref name="n"/> values; returns nothing, the row sums to 1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SoftmaxInPlace(float* x, int n)
    {
        int i = 0;
        float max = float.NegativeInfinity;
        if (UseAvx512 && n >= 16)
        {
            var vm = Vector512.Create(float.NegativeInfinity);
            for (; i + 16 <= n; i += 16) vm = Vector512.Max(vm, Vector512.Load(x + i));
            for (int l = 0; l < 16; l++) max = MathF.Max(max, vm.GetElement(l));
        }
        for (; i < n; i++) max = MathF.Max(max, x[i]);
        float sum = 0;
        i = 0;
        if (UseAvx512)
        {
            var m = Vector512.Create(max);
            var s = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var e = VMath.Exp(Vector512.Load(x + i) - m);
                Vector512.Store(e, x + i);
                s += e;
            }
            sum = Vector512.Sum(s);
        }
        for (; i < n; i++) { float e = MathF.Exp(x[i] - max); x[i] = e; sum += e; }
        float inv = 1f / sum;
        i = 0;
        if (UseAvx512)
        {
            var vi = Vector512.Create(inv);
            for (; i + 16 <= n; i += 16) Vector512.Store(Vector512.Load(x + i) * vi, x + i);
        }
        for (; i < n; i++) x[i] *= inv;
    }

    /// <summary>Transposes a row-major <c>[rows, cols]</c> matrix into <c>[cols, rows]</c>.</summary>
    public static void Transpose(float* src, float* dst, int rows, int cols)
    {
        const int B = 32;
        for (int r0 = 0; r0 < rows; r0 += B)
            for (int c0 = 0; c0 < cols; c0 += B)
                for (int r = r0; r < Math.Min(rows, r0 + B); r++)
                    for (int c = c0; c < Math.Min(cols, c0 + B); c++)
                        dst[(long)c * rows + r] = src[(long)r * cols + c];
    }
}
