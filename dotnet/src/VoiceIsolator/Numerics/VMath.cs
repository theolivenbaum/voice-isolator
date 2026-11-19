using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace VoiceIsolation.Numerics;

/// <summary>Vectorized transcendental functions used inside fused kernels (GEMM epilogues, softmax, the selective scan).
/// Adapted from curiosity-ai/parakeet (Apache-2.0); <see cref="Erf(Vector512{float})"/>, <see cref="Gelu(Vector512{float})"/>
/// and <see cref="Softplus(Vector512{float})"/> added.
/// <c>exp</c> uses the Cephes range reduction <c>e^x = 2^n · e^r</c>, |r| ≤ ln2/2, with a degree-6 polynomial
/// (≈1 ulp over the float range), so SiLU/sigmoid match a libm-based reference to float precision.</summary>
public static class VMath
{
    private const float Log2E = 1.44269504088896341f;
    private const float Ln2Hi = 0.693359375f;
    private const float Ln2Lo = -2.12194440e-4f;
    private const float ExpMax = 88.3762626647949f;
    private const float ExpMin = -87.3365447505531f;
    private const float P0 = 1.9875691500E-4f, P1 = 1.3981999507E-3f, P2 = 8.3334519073E-3f;
    private const float P3 = 4.1665795894E-2f, P4 = 1.6666665459E-1f, P5 = 5.0000001201E-1f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        x = Vector512.Min(Vector512.Max(x, Vector512.Create(ExpMin)), Vector512.Create(ExpMax));
        var n = Avx512F.RoundScale(x * Vector512.Create(Log2E), 0x08);   // round to nearest, no exceptions
        var r = Avx512F.FusedMultiplyAddNegated(n, Vector512.Create(Ln2Hi), x);
        r = Avx512F.FusedMultiplyAddNegated(n, Vector512.Create(Ln2Lo), r);
        var p = Avx512F.FusedMultiplyAdd(Vector512.Create(P0), r, Vector512.Create(P1));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P2));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P3));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P4));
        p = Avx512F.FusedMultiplyAdd(p, r, Vector512.Create(P5));
        p = Avx512F.FusedMultiplyAdd(p, r * r, r + Vector512<float>.One);
        return Avx512F.Scale(p, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        x = Vector256.Min(Vector256.Max(x, Vector256.Create(ExpMin)), Vector256.Create(ExpMax));
        var n = Avx.RoundToNearestInteger(x * Vector256.Create(Log2E));
        var r = Fma.MultiplyAddNegated(n, Vector256.Create(Ln2Hi), x);
        r = Fma.MultiplyAddNegated(n, Vector256.Create(Ln2Lo), r);
        var p = Fma.MultiplyAdd(Vector256.Create(P0), r, Vector256.Create(P1));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P2));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P3));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P4));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(P5));
        p = Fma.MultiplyAdd(p, r * r, r + Vector256<float>.One);
        var pow2 = Avx2.ShiftLeftLogical(Avx.ConvertToVector256Int32(n) + Vector256.Create(127), 23).AsSingle();
        return p * pow2;
    }

    /// <summary>x · sigmoid(x) (Swish / SiLU).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Silu(Vector512<float> x) => x / (Vector512<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Silu(Vector256<float> x) => x / (Vector256<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Sigmoid(Vector512<float> x) => Vector512<float>.One / (Vector512<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x) => Vector256<float>.One / (Vector256<float>.One + Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    // erf: Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7, evaluated exactly as PyTorch's vectorized CPU erf (the
    // kernel behind torch.nn.GELU on the CPU), so GELU matches the reference to float rounding.
    private const float ErfP = 0.3275911f, ErfA1 = 0.254829592f, ErfA2 = -0.284496736f, ErfA3 = 1.421413741f;
    private const float ErfA4 = -1.453152027f, ErfA5 = 1.061405429f;
    private const float Sqrt1_2 = 0.707106781186547524f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Erf(Vector512<float> x)
    {
        var sign = x & Vector512.Create(-0f);
        var ax = Vector512.Abs(x);
        var one = Vector512<float>.One;
        var t = one / Vector512.FusedMultiplyAdd(Vector512.Create(ErfP), ax, one);
        var r = Vector512.FusedMultiplyAdd(Vector512.Create(ErfA5), t, Vector512.Create(ErfA4));
        r = Vector512.FusedMultiplyAdd(r, t, Vector512.Create(ErfA3));
        r = Vector512.FusedMultiplyAdd(r, t, Vector512.Create(ErfA2));
        r = Vector512.FusedMultiplyAdd(r, t, Vector512.Create(ErfA1));
        var e = -Exp(-(x * x));
        return Vector512.FusedMultiplyAdd(e * t, r, one) ^ sign;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Erf(Vector256<float> x)
    {
        var sign = x & Vector256.Create(-0f);
        var ax = Vector256.Abs(x);
        var one = Vector256<float>.One;
        var t = one / Vector256.FusedMultiplyAdd(Vector256.Create(ErfP), ax, one);
        var r = Vector256.FusedMultiplyAdd(Vector256.Create(ErfA5), t, Vector256.Create(ErfA4));
        r = Vector256.FusedMultiplyAdd(r, t, Vector256.Create(ErfA3));
        r = Vector256.FusedMultiplyAdd(r, t, Vector256.Create(ErfA2));
        r = Vector256.FusedMultiplyAdd(r, t, Vector256.Create(ErfA1));
        var e = -Exp(-(x * x));
        return Vector256.FusedMultiplyAdd(e * t, r, one) ^ sign;
    }

    public static float Erf(float x)
    {
        float ax = MathF.Abs(x);
        float t = 1f / MathF.FusedMultiplyAdd(ErfP, ax, 1f);
        float r = MathF.FusedMultiplyAdd(ErfA5, t, ErfA4);
        r = MathF.FusedMultiplyAdd(r, t, ErfA3);
        r = MathF.FusedMultiplyAdd(r, t, ErfA2);
        r = MathF.FusedMultiplyAdd(r, t, ErfA1);
        float y = MathF.FusedMultiplyAdd(-MathF.Exp(-(x * x)) * t, r, 1f);
        return MathF.CopySign(y, x);
    }

    /// <summary>The exact GELU, <c>x/2 · (1 + erf(x/√2))</c> (torch.nn.GELU()).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Gelu(Vector512<float> x) =>
        x * Vector512.Create(0.5f) * (Vector512<float>.One + Erf(x * Vector512.Create(Sqrt1_2)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Gelu(Vector256<float> x) =>
        x * Vector256.Create(0.5f) * (Vector256<float>.One + Erf(x * Vector256.Create(Sqrt1_2)));

    public static float Gelu(float x) => x * 0.5f * (1f + Erf(x * Sqrt1_2));

    /// <summary><c>log(1 + e^x)</c>, and <c>x</c> above 20 (torch.nn.functional.softplus).</summary>
    public static float Softplus(float x) => x > 20f ? x : (float)Math.Log(1.0 + Math.Exp(x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Softplus(Vector512<float> x)
    {
        // log1p(e^x) = log(u) - ((u - 1) - e^x) / u with u = 1 + e^x (Goldberg's compensated log1p).
        var e = Exp(x);
        var u = Vector512<float>.One + e;
        var y = Log(u) - ((u - Vector512<float>.One) - e) / u;
        return Vector512.ConditionalSelect(Vector512.GreaterThan(x, Vector512.Create(20f)), x, y);
    }

    /// <summary>Natural logarithm for positive normal inputs (Cephes logf, ≈1 ulp).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Log(Vector512<float> x)
    {
        var bits = x.AsInt32();
        var exponent = Vector512.ConvertToSingle((bits >>> 23) - Vector512.Create(126));
        var m = ((bits & Vector512.Create(0x007FFFFF)) | Vector512.Create(0x3F000000)).AsSingle();   // [0.5, 1)
        var small = Vector512.LessThan(m, Vector512.Create(0.707106781186547524f));
        exponent -= Vector512.ConditionalSelect(small, Vector512<float>.One, Vector512<float>.Zero);
        m = Vector512.ConditionalSelect(small, m + m, m) - Vector512<float>.One;
        var z = m * m;
        var p = Vector512.Create(7.0376836292E-2f);
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(-1.1514610310E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(1.1676998740E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(-1.2420140846E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(1.4249322787E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(-1.6668057665E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(2.0000714765E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(-2.4999993993E-1f));
        p = Vector512.FusedMultiplyAdd(p, m, Vector512.Create(3.3333331174E-1f));
        var y = p * m * z;
        y = Vector512.FusedMultiplyAdd(exponent, Vector512.Create(-2.12194440e-4f), y);
        y = Vector512.FusedMultiplyAdd(z, Vector512.Create(-0.5f), y);
        return Vector512.FusedMultiplyAdd(exponent, Vector512.Create(0.693359375f), m + y);
    }
}
