#!/usr/bin/env python3
"""Generates the unrolled micro-kernels of src/VoiceIsolator/Numerics/Gemm.cs (C# has no macros or register templates).

Adapted from curiosity-ai/parakeet (scripts/gen_kernels.py, Apache-2.0), with a GELU epilogue added.

  python scripts/gen_kernels.py fp32   # Kernel512Rows + Kernel512R1..R12 for Gemm.cs (AVX-512, 12x32 tile)

Paste the output over the corresponding block of the target file.
"""
import sys


def fp32(MR=12):
    L = []
    w = L.append
    w("    /// <summary>Dispatches the AVX-512 micro-kernel for a tile of <paramref name=\"rows\"/> (1-12) rows × 32 columns.</summary>")
    w("    [MethodImpl(MethodImplOptions.AggressiveInlining)]")
    w("    private static void Kernel512Rows(int rows, float* a, float* b, int kc, float* c, long ldc, int mode, float* bias, Activation act)")
    w("    {")
    w("        switch (rows)")
    w("        {")
    for r in range(1, MR + 1):
        w(f"            case {r}: Kernel512R{r}(a, b, kc, c, ldc, mode, bias, act); break;")
    w("        }")
    w("    }")
    w("")
    for R in range(1, MR + 1):
        w("    [MethodImpl(MethodImplOptions.AggressiveOptimization)]")
        w(f"    private static void Kernel512R{R}(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias, Activation act)")
        w("    {")
        w("        " + "Vector512<float> " + ", ".join(f"c{i}a, c{i}b" for i in range(R)) + ";")
        w("        if (mode == InitZero)")
        w("        {")
        for i in range(R): w(f"            c{i}a = Vector512<float>.Zero; c{i}b = Vector512<float>.Zero;")
        w("        }")
        w("        else if (mode == InitBias)")
        w("        {")
        w("            var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);")
        for i in range(R): w(f"            c{i}a = ba; c{i}b = bb;")
        w("        }")
        w("        else")
        w("        {")
        for i in range(R): w(f"            c{i}a = Vector512.Load(c + {i} * ldc); c{i}b = Vector512.Load(c + {i} * ldc + 16);")
        w("            if (mode == InitLoadBias)")
        w("            {")
        w("                var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);")
        for i in range(R): w(f"                c{i}a += ba; c{i}b += bb;")
        w("            }")
        w("        }")
        w("        for (int p = 0; p < kc; p++)")
        w("        {")
        w("            Sse.Prefetch0(b + PrefetchDistance);")
        w("            Sse.Prefetch0(b + PrefetchDistance + 16);")
        w("            var b0 = Vector512.Load(b); var b1 = Vector512.Load(b + 16);")
        for i in range(R):
            w(f"            {{ var x = Vector512.Create(a[{i}]); c{i}a = Avx512F.FusedMultiplyAdd(x, b0, c{i}a); c{i}b = Avx512F.FusedMultiplyAdd(x, b1, c{i}b); }}")
        w(f"            a += {MR}; b += 32;")
        w("        }")
        w("        if (act == Activation.Silu)")
        w("        {")
        for i in range(R): w(f"            c{i}a = VMath.Silu(c{i}a); c{i}b = VMath.Silu(c{i}b);")
        w("        }")
        w("        else if (act == Activation.Gelu)")
        w("        {")
        for i in range(R): w(f"            c{i}a = VMath.Gelu(c{i}a); c{i}b = VMath.Gelu(c{i}b);")
        w("        }")
        w("        else if (act == Activation.Relu)")
        w("        {")
        w("            var z = Vector512<float>.Zero;")
        for i in range(R): w(f"            c{i}a = Vector512.Max(c{i}a, z); c{i}b = Vector512.Max(c{i}b, z);")
        w("        }")
        for i in range(R): w(f"        Vector512.Store(c{i}a, c + {i} * ldc); Vector512.Store(c{i}b, c + {i} * ldc + 16);")
        w("    }")
        w("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "fp32":
        sys.stdout.write(fp32())
    else:
        sys.exit(__doc__)
