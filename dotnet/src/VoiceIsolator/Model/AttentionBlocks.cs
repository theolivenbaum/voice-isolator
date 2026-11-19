using VoiceIsolation.IO;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>
/// A pre-norm transformer layer along time (<c>_TimeBlock</c>): rotary self attention within each band, the exclusive
/// projection (each token's output loses its component along its own value), then a GELU feed-forward.
/// </summary>
internal sealed unsafe class TimeTransformer : IDisposable
{
    private const int RowBlock = 64;
    private const int TargetRows = 8192;
    private const int FfnRows = 2048;
    private const int NR = PackedMatrix.NR;

    [ThreadStatic] private static NativeBuffer<float>? t_scores;

    public TimeTransformer(SafeTensorsFile f, string prefix, VoiceIsolatorConfig config, bool exclusive)
    {
        Width = config.Width;
        Heads = config.NumHeads;
        Exclusive = exclusive;
        Window = config.AttentionWindowFrames;
        Norm1 = Norm.Load(f, prefix + ".norm1", Width);
        Qkv = Linear.Load(f, prefix + ".attn.qkv", 3 * Width, Width);
        Proj = Linear.Load(f, prefix + ".attn.proj", Width, Width);
        Norm2 = Norm.Load(f, prefix + ".norm2", Width);
        Ffn1 = Linear.Load(f, prefix + ".ffn.0", 4 * Width, Width);
        Ffn2 = Linear.Load(f, prefix + ".ffn.3", Width, 4 * Width);
    }

    public int Width { get; }
    public int Heads { get; }
    public int HeadDim => Width / Heads;
    public bool Exclusive { get; }
    public int Window { get; }
    public Norm Norm1 { get; }
    public Linear Qkv { get; }
    public Linear Proj { get; }
    public Norm Norm2 { get; }
    public Linear Ffn1 { get; }
    public Linear Ffn2 { get; }

    /// <summary>Runs the layer in place over <paramref name="x"/> <c>[sequences, frames, width]</c>.</summary>
    public void Forward(float* x, int sequences, int frames, bool causal, Workspace ws, Parallelism par)
    {
        int W = Width;
        int group = ws.Trace != null ? sequences : Math.Clamp(TargetRows / frames, 1, sequences);
        for (int s0 = 0; s0 < sequences; s0 += group)
        {
            int g = Math.Min(group, sequences - s0), rows = g * frames;
            float* xg = x + (long)s0 * frames * W;
            float* h = ws.Get(Slot.Norm, (long)rows * W);
            float* qkv = ws.Get(Slot.Qkv, (long)rows * 3 * W);
            Norm1.Forward(xg, h, rows, par);
            Qkv.Forward(h, W, rows, qkv, 3 * W, par);
            ApplyRope(qkv, rows, frames, 0, par);
            float* context = h;
            Attend(qkv, g, frames, causal, context, ws, par);
            if (ws.Trace != null)
            {
                float* attn = ws.Get(Slot.Residual, (long)rows * W);
                Proj.Forward(context, W, rows, attn, W, par);
                ws.Emit("time.attn", attn, (long)rows * W);
            }
            Proj.Forward(context, W, rows, xg, W, par, accumulate: true);
        }
        FeedForward(x, (long)sequences * frames, ws, par);
    }

    /// <summary><c>x += FFN(LN2(x))</c> over <paramref name="rows"/> rows, a slice of rows at a time.</summary>
    public void FeedForward(float* x, long rows, Workspace ws, Parallelism par)
    {
        int W = Width;
        float* h = ws.Get(Slot.Norm, (long)FfnRows * W);
        float* hidden = ws.Get(Slot.Hidden, (long)FfnRows * 4 * W);
        for (long r0 = 0; r0 < rows; r0 += FfnRows)
        {
            int n = (int)Math.Min(FfnRows, rows - r0);
            float* xr = x + r0 * W;
            Norm2.Forward(xr, h, n, par);
            Ffn1.Forward(h, W, n, hidden, 4 * W, par, Activation.Gelu);
            Ffn2.Forward(hidden, 4 * W, n, xr, W, par, accumulate: true);
        }
    }

    /// <summary>Rotates the queries and keys of every row in place; row <c>r</c> is at position
    /// <c>first + r % frames</c>.</summary>
    public void ApplyRope(float* qkv, int rows, int frames, int first, Parallelism par)
    {
        int W = Width, hd = HeadDim, heads = Heads;
        var (cos, sin) = Rope.Tables(first + frames, hd);
        par.ForRange(rows, 64, (r0, r1) =>
        {
            for (int r = r0; r < r1; r++)
            {
                float* row = qkv + (long)r * 3 * W;
                int position = first + r % frames;
                for (int head = 0; head < heads; head++)
                {
                    Rope.Apply(row + head * hd, hd, cos, sin, position);
                    Rope.Apply(row + W + head * hd, hd, cos, sin, position);
                }
            }
        });
    }

    /// <summary>
    /// Scaled dot-product attention of every (sequence, head) over its frames, writing the head-concatenated context
    /// <c>[sequences·frames, width]</c>. Keys and values are packed per head as GEMM operands, then blocks of 64 query rows
    /// run score GEMM, mask, softmax and value GEMM back to back while their rows are cache-hot. Causal attention may be
    /// bounded to <see cref="Window"/> frames (each frame sees itself and <c>window - 1</c> before it).
    /// </summary>
    private void Attend(float* qkv, int sequences, int frames, bool causal, float* context, Workspace ws, Parallelism par)
    {
        int W = Width, hd = HeadDim, heads = Heads;
        long ld = 3L * W;
        long kSize = PackedMatrix.SizeOf(hd, frames), vSize = PackedMatrix.SizeOf(frames, hd);
        long perHead = kSize + vSize;
        float* shared = ws.Get(Slot.AttentionPack, sequences * heads * perHead);
        par.For(sequences * heads, item =>
        {
            int s = item / heads, head = item % heads;
            float* rows = qkv + (long)s * frames * ld;
            float* dst = shared + item * perHead;
            PackedMatrix.PackLinear(rows + W + head * hd, frames, hd, ld, dst);
            PackedMatrix.PackKN(rows + 2 * W + head * hd, frames, hd, ld, dst + kSize);
        });

        int blocks = (frames + RowBlock - 1) / RowBlock;
        float scale = 1f / MathF.Sqrt(hd);
        int window = causal ? Window : 0;
        bool exclusive = Exclusive;
        par.For(sequences * heads * blocks, item =>
        {
            int pair = item / blocks, blk = item % blocks;
            int s = pair / heads, head = pair % heads;
            int i0 = blk * RowBlock, i1 = Math.Min(frames, i0 + RowBlock), rows = i1 - i0;
            float* basePtr = shared + pair * perHead;
            var kMat = new PackedMatrix(basePtr, hd, frames);
            var vMat = new PackedMatrix(basePtr + kSize, frames, hd);
            int lo0 = window > 0 ? Math.Max(0, i0 - window + 1) : 0;
            int hiEnd = causal ? i1 : frames;
            int nb0 = lo0 / NR, nb1 = (hiEnd + NR - 1) / NR;

            long need = (long)rows * frames;
            var buf = t_scores;
            if (buf == null || buf.Length < need)
            {
                buf?.Dispose();
                buf = t_scores = new NativeBuffer<float>(Math.Max(need, 1 << 16), zero: false);
            }
            float* scores = buf.Ptr;
            float* q = qkv + ((long)s * frames + i0) * ld + head * hd;
            Gemm.Multiply(q, ld, rows, kMat, nb0, nb1, scores + nb0 * NR, frames, Parallelism.Single);
            for (int r = 0; r < rows; r++)
            {
                int i = i0 + r;
                int lo = window > 0 ? Math.Max(0, i - window + 1) : 0;
                int hi = causal ? i + 1 : frames;
                float* row = scores + (long)r * frames;
                var span = new Span<float>(row, frames);
                span[..lo].Clear();
                span[hi..].Clear();
                var valid = span[lo..hi];
                System.Numerics.Tensors.TensorPrimitives.Multiply(valid, scale, valid);
                Ops.SoftmaxInPlace(row + lo, hi - lo);
            }
            float* outRows = context + ((long)s * frames + i0) * W + head * hd;
            Gemm.Multiply(scores, frames, rows, vMat, outRows, W, Parallelism.Single);
            if (exclusive)
            {
                for (int r = 0; r < rows; r++)
                    ExclusiveProjection(outRows + (long)r * W, qkv + ((long)s * frames + i0 + r) * ld + 2 * W + head * hd, hd);
            }
        });
    }

    /// <summary><c>o -= (o·v / max(v·v, 1e-6)) v</c> (<c>model.exclusive</c>).</summary>
    public static void ExclusiveProjection(float* o, float* v, int n)
    {
        var os = new Span<float>(o, n);
        var vs = new ReadOnlySpan<float>(v, n);
        float coefficient = System.Numerics.Tensors.TensorPrimitives.Dot(os, vs) /
                            MathF.Max(System.Numerics.Tensors.TensorPrimitives.Dot(vs, vs), 1e-6f);
        // o - c·v, rounded like the reference (product, then difference)
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var c = Vector512.Create(coefficient);
            for (; i + 16 <= n; i += 16) Vector512.Store(Vector512.Load(o + i) - c * Vector512.Load(v + i), o + i);
        }
        for (; i < n; i++) o[i] -= coefficient * v[i];
    }

    public void Dispose()
    {
        Norm1.Dispose();
        Qkv.Dispose();
        Proj.Dispose();
        Norm2.Dispose();
        Ffn1.Dispose();
        Ffn2.Dispose();
    }
}

/// <summary>A pre-norm transformer layer across the bands of each frame (<c>_BandTransformer</c>): plain multi-head self
/// attention (no positions) over the 46 band tokens of a frame, then a GELU feed-forward.</summary>
internal sealed unsafe class BandTransformer : IDisposable
{
    private const int TargetRows = 8192;
    private const int FfnRows = 2048;

    public BandTransformer(SafeTensorsFile f, string prefix, VoiceIsolatorConfig config, bool exclusive)
    {
        Width = config.Width;
        Heads = config.NumHeads;
        Exclusive = exclusive;
        Norm1 = Norm.Load(f, prefix + ".norm1", Width);
        InProj = new Linear(f.Float(prefix + ".self_attn.in_proj_weight", 3 * Width, Width), 3 * Width, Width,
                            f.Float(prefix + ".self_attn.in_proj_bias", 3 * Width));
        OutProj = Linear.Load(f, prefix + ".self_attn.out_proj", Width, Width);
        Norm2 = Norm.Load(f, prefix + ".norm2", Width);
        Linear1 = Linear.Load(f, prefix + ".linear1", 4 * Width, Width);
        Linear2 = Linear.Load(f, prefix + ".linear2", Width, 4 * Width);
    }

    public int Width { get; }
    public int Heads { get; }
    public bool Exclusive { get; }
    public Norm Norm1 { get; }
    public Linear InProj { get; }
    public Linear OutProj { get; }
    public Norm Norm2 { get; }
    public Linear Linear1 { get; }
    public Linear Linear2 { get; }

    /// <summary>Runs the layer in place over <paramref name="x"/> <c>[frames, bands, width]</c>.</summary>
    public void Forward(float* x, int frames, int bands, Workspace ws, Parallelism par)
    {
        int W = Width;
        int group = ws.Trace != null ? frames : Math.Clamp(TargetRows / bands, 1, frames);
        for (int f0 = 0; f0 < frames; f0 += group)
        {
            int g = Math.Min(group, frames - f0), rows = g * bands;
            float* xg = x + (long)f0 * bands * W;
            float* h = ws.Get(Slot.Norm, (long)rows * W);
            float* qkv = ws.Get(Slot.Qkv, (long)rows * 3 * W);
            Norm1.Forward(xg, h, rows, par);
            InProj.Forward(h, W, rows, qkv, 3 * W, par);
            float* context = h;
            int heads = Heads, hd = W / heads;
            bool exclusive = Exclusive;
            par.For(g, frame =>
            {
                float* scores = stackalloc float[bands];
                for (int head = 0; head < heads; head++)
                {
                    float* q0 = qkv + (long)frame * bands * 3 * W + head * hd;
                    for (int i = 0; i < bands; i++)
                    {
                        float* o = context + ((long)frame * bands + i) * W + head * hd;
                        AttendRow(q0 + (long)i * 3 * W, q0 + W, q0 + 2 * W, 3 * W, bands, hd, scores, o);
                        if (exclusive) TimeTransformer.ExclusiveProjection(o, q0 + (long)i * 3 * W + 2 * W, hd);
                    }
                }
            });
            OutProj.Forward(context, W, rows, xg, W, par, accumulate: true);
        }
        float* hn = ws.Get(Slot.Norm, (long)FfnRows * W);
        float* hidden = ws.Get(Slot.Hidden, (long)FfnRows * 4 * W);
        long total = (long)frames * bands;
        for (long r0 = 0; r0 < total; r0 += FfnRows)
        {
            int n = (int)Math.Min(FfnRows, total - r0);
            float* xr = x + r0 * W;
            Norm2.Forward(xr, hn, n, par);
            Linear1.Forward(hn, W, n, hidden, 4 * W, par, Activation.Gelu);
            Linear2.Forward(hidden, 4 * W, n, xr, W, par, accumulate: true);
        }
    }

    /// <summary>One query's attention over <paramref name="n"/> keys and values (rows <paramref name="ld"/> apart):
    /// <c>o = softmax(q·Kᵀ / √d) · V</c>.</summary>
    internal static void AttendRow(float* q, float* k, float* v, long ld, int n, int hd, float* scores, float* o)
    {
        float scale = 1f / MathF.Sqrt(hd);
        for (int j = 0; j < n; j++) scores[j] = Dot(q, k + j * ld, hd) * scale;
        Ops.SoftmaxInPlace(scores, n);
        Accumulate(scores, v, ld, n, hd, o);
    }

    /// <summary><c>o = Σ_j p_j · v_j</c> over <paramref name="n"/> value rows.</summary>
    internal static void Accumulate(float* p, float* v, long ld, int n, int hd, float* o)
    {
        int c = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (; c + 16 <= hd; c += 16)
            {
                var acc = Vector512<float>.Zero;
                for (int j = 0; j < n; j++) acc = Vector512.FusedMultiplyAdd(Vector512.Create(p[j]), Vector512.Load(v + j * ld + c), acc);
                Vector512.Store(acc, o + c);
            }
        }
        for (; c < hd; c++)
        {
            float acc = 0;
            for (int j = 0; j < n; j++) acc = MathF.FusedMultiplyAdd(p[j], v[j * ld + c], acc);
            o[c] = acc;
        }
    }

    internal static float Dot(float* a, float* b, int n)
    {
        int i = 0;
        float sum = 0;
        if (Vector512.IsHardwareAccelerated && n >= 16)
        {
            var acc = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16) acc = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), acc);
            sum = Vector512.Sum(acc);
        }
        for (; i < n; i++) sum = MathF.FusedMultiplyAdd(a[i], b[i], sum);
        return sum;
    }

    public void Dispose()
    {
        Norm1.Dispose();
        InProj.Dispose();
        OutProj.Dispose();
        Norm2.Dispose();
        Linear1.Dispose();
        Linear2.Dispose();
    }
}
