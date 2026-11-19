using System.Numerics.Tensors;
using VoiceIsolation.IO;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>
/// The state a stream carries through one causal Mamba layer for its rows (sequences): the last <c>conv - 1</c> inputs of
/// the depthwise convolution, <c>[rows, conv - 1, inner]</c> oldest first, and the scan state <c>[rows, state, inner]</c>.
/// </summary>
internal sealed unsafe class MambaState : IDisposable
{
    private readonly NativeBuffer<float> _tail;
    private readonly NativeBuffer<float> _ssm;

    public MambaState(int rows, int taps, int states, int inner)
    {
        Rows = rows;
        Taps = taps;
        States = states;
        Inner = inner;
        _tail = new NativeBuffer<float>((long)rows * Math.Max(taps, 1) * inner, zero: true, hugePages: false);
        _ssm = new NativeBuffer<float>((long)rows * states * inner, zero: true, hugePages: false);
    }

    public int Rows { get; }
    public int Taps { get; }
    public int States { get; }
    public int Inner { get; }
    public float* Tail => _tail.Ptr;
    public float* Ssm => _ssm.Ptr;

    public MambaState Clone()
    {
        var copy = new MambaState(Rows, Taps, States, Inner);
        _tail.AsSpan().CopyTo(copy._tail.AsSpan());
        _ssm.AsSpan().CopyTo(copy._ssm.AsSpan());
        return copy;
    }

    public void Dispose()
    {
        _tail.Dispose();
        _ssm.Dispose();
    }
}

/// <summary>
/// One selective state-space layer (<c>model.Mamba</c>): input projection to <c>u</c> and gate <c>z</c>, a depthwise causal
/// convolution and SiLU over <c>u</c>, the data-dependent step <c>Δ = softplus(W_dt·dt + b)</c> and projections <c>B</c>,
/// <c>C</c>, the scan <c>h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t</c>, <c>y_t = C_t·h_t + D u_t</c>, gated by SiLU(z),
/// and the output projection. A reverse run is the layer over the time-flipped sequence, flipped back, without copies.
/// </summary>
internal sealed unsafe class MambaLayer : IDisposable
{
    public MambaLayer(SafeTensorsFile f, string prefix, VoiceIsolatorConfig config)
    {
        Width = config.Width;
        Inner = config.MambaInner;
        States = config.MambaState;
        Taps = config.MambaConv;
        DtRank = config.DtRank;
        InProj = Linear.Load(f, prefix + ".in_proj", 2 * Inner, Width, bias: false);
        XProj = Linear.Load(f, prefix + ".x_proj", DtRank + 2 * States, Inner, bias: false);
        DtProj = Linear.Load(f, prefix + ".dt_proj", Inner, DtRank);
        OutProj = Linear.Load(f, prefix + ".out_proj", Width, Inner, bias: false);

        // Depthwise taps, tap-major [taps, inner], so a vector of channels shares one load per tap.
        float* w = f.Float(prefix + ".conv1d.weight", Inner, 1, Taps);
        ConvWeight = new Vec(Taps * Inner);
        for (int c = 0; c < Inner; c++)
            for (int k = 0; k < Taps; k++) ConvWeight.Ptr[k * Inner + c] = w[c * Taps + k];
        ConvBias = new Vec(new ReadOnlySpan<float>(f.Float(prefix + ".conv1d.bias", Inner), Inner));

        // A = -exp(A_log) in float32, state-major [states, inner].
        float* aLog = f.Float(prefix + ".A_log", Inner, States);
        A = new Vec(States * Inner);
        for (int c = 0; c < Inner; c++)
            for (int n = 0; n < States; n++) A.Ptr[n * Inner + c] = -MathF.Exp(aLog[c * States + n]);
        D = new Vec(new ReadOnlySpan<float>(f.Float(prefix + ".D", Inner), Inner));
    }

    public int Width { get; }
    public int Inner { get; }
    public int States { get; }
    public int Taps { get; }
    public int DtRank { get; }
    public Linear InProj { get; }
    public Linear XProj { get; }
    public Linear DtProj { get; }
    public Linear OutProj { get; }
    public Vec ConvWeight { get; }
    public Vec ConvBias { get; }
    public Vec A { get; }
    public Vec D { get; }

    /// <summary>
    /// Runs the layer over <paramref name="x"/> <c>[sequences, length, width]</c> and writes (or, when
    /// <paramref name="accumulate"/>, adds) its output to <paramref name="y"/>, rows <paramref name="ldy"/> apart.
    /// <paramref name="state"/> (forward runs only) carries the convolution tail and scan state of each sequence in and out.
    /// </summary>
    public void Forward(float* x, int sequences, int length, bool reverse, float* y, long ldy, bool accumulate,
                        MambaState? state, Workspace ws, Parallelism par)
    {
        int rows = sequences * length, inner = Inner, xw = DtRank + 2 * States;
        float* xz = ws.Get(Slot.MambaXz, (long)rows * 2 * inner);
        float* u = ws.Get(Slot.MambaU, (long)rows * inner);
        float* xdbl = ws.Get(Slot.MambaXdbl, (long)rows * xw);
        float* delta = ws.Get(Slot.MambaDelta, (long)rows * inner);
        float* scanOut = ws.Get(Slot.MambaY, (long)rows * inner);

        InProj.Forward(x, Width, rows, xz, 2 * inner, par);
        Convolve(xz, sequences, length, reverse, u, state, par);
        XProj.Forward(u, inner, rows, xdbl, xw, par);
        DtProj.Forward(xdbl, xw, rows, delta, inner, par);
        par.ForRange(rows, 64, (r0, r1) =>
        {
            var span = new Span<float>(delta + (long)r0 * inner, (r1 - r0) * inner);
            Softplus(span);
        });
        Scan(u, delta, xdbl, xz, sequences, length, reverse, scanOut, state, par);
        OutProj.Forward(scanOut, inner, rows, y, ldy, par, accumulate: accumulate);
    }

    /// <summary>Δ = softplus(x) in place: <c>log(1 + e^x)</c>, and <c>x</c> above 20.</summary>
    internal static void Softplus(Span<float> x)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            ref float r = ref x[0];
            for (; i + 16 <= x.Length; i += 16)
            {
                var v = Vector512.LoadUnsafe(ref r, (nuint)i);
                VMath.Softplus(v).StoreUnsafe(ref r, (nuint)i);
            }
        }
        for (; i < x.Length; i++) x[i] = VMath.Softplus(x[i]);
    }

    /// <summary>
    /// <c>u[t] = SiLU(b + Σ_k w_k · u_in[t + k - (taps - 1)])</c> along each sequence (zeros, or the carried tail, before its
    /// start); a reverse run reads <c>u_in[t - k + (taps - 1)]</c> (zeros after the end). <c>u_in</c> is the first half of
    /// <paramref name="xz"/>'s rows. A carried tail is updated to the last <c>taps - 1</c> inputs.
    /// </summary>
    private void Convolve(float* xz, int sequences, int length, bool reverse, float* u, MambaState? state, Parallelism par)
    {
        int inner = Inner, taps = Taps, pad = taps - 1;
        long ldx = 2L * inner;
        float* w = ConvWeight.Ptr, bias = ConvBias.Ptr;
        int rows = sequences * length;
        par.ForRange(rows, 16, (r0, r1) =>
        {
            float** src = stackalloc float*[taps];
            for (int r = r0; r < r1; r++)
            {
                int s = r / length, t = r % length;
                for (int k = 0; k < taps; k++)
                {
                    int at = reverse ? t - k + pad : t + k - pad;
                    if (at >= 0 && at < length) src[k] = xz + ((long)s * length + at) * ldx;
                    else if (!reverse && state != null && at < 0) src[k] = state.Tail + ((long)s * pad + (pad + at)) * inner;
                    else src[k] = null;
                }
                float* o = u + (long)r * inner;
                int c = 0;
                if (Vector512.IsHardwareAccelerated)
                {
                    for (; c + 16 <= inner; c += 16)
                    {
                        var acc = Vector512.Load(bias + c);
                        for (int k = 0; k < taps; k++)
                            if (src[k] != null) acc = Vector512.FusedMultiplyAdd(Vector512.Load(w + k * inner + c), Vector512.Load(src[k] + c), acc);
                        Vector512.Store(VMath.Silu(acc), o + c);
                    }
                }
                for (; c < inner; c++)
                {
                    float acc = bias[c];
                    for (int k = 0; k < taps; k++)
                        if (src[k] != null) acc = MathF.FusedMultiplyAdd(w[k * inner + c], src[k][c], acc);
                    o[c] = VMath.Silu(acc);
                }
            }
        });
        if (state != null && pad > 0)
        {
            // The new tail: the last `pad` inputs of each sequence, older carried ones first when the block is short.
            for (int s = 0; s < sequences; s++)
            {
                float* tail = state.Tail + (long)s * pad * inner;
                int keep = Math.Max(0, pad - length);
                if (keep > 0) Buffer.MemoryCopy(tail + (pad - keep) * inner, tail, (long)keep * inner * 4, (long)keep * inner * 4);
                for (int j = keep; j < pad; j++)
                {
                    int t = length - (pad - j);
                    Buffer.MemoryCopy(xz + ((long)s * length + t) * ldx, tail + (long)j * inner, inner * 4L, inner * 4L);
                }
            }
        }
    }

    /// <summary>
    /// The selective scan, in float32, each work item one sequence and 16 channels with their states in a small
    /// stack buffer: per frame, 16 vector exponentials <c>exp(Δ·A_n)</c>, the state update and the readout against
    /// <c>C_t</c>, then the skip <c>D·u</c> and the SiLU gate.
    /// </summary>
    private void Scan(float* u, float* delta, float* xdbl, float* xz, int sequences, int length, bool reverse, float* y,
                      MambaState? state, Parallelism par)
    {
        int inner = Inner, states = States, xw = DtRank + 2 * States, bOff = DtRank, cOff = DtRank + States;
        int groups = inner / 16;
        float* a = A.Ptr, d = D.Ptr;
        long ldz = 2L * inner;
        par.For(sequences * groups, item =>
        {
            int s = item / groups, c0 = (item % groups) * 16;
            long baseRow = (long)s * length;
            float* hState = state == null ? null : state.Ssm + (long)s * states * inner + c0;
            if (Vector512.IsHardwareAccelerated)
            {
                var h = stackalloc Vector512<float>[states];
                for (int n = 0; n < states; n++) h[n] = hState == null ? Vector512<float>.Zero : Vector512.Load(hState + (long)n * inner);
                var dv = Vector512.Load(d + c0);
                for (int step = 0; step < length; step++)
                {
                    int t = reverse ? length - 1 - step : step;
                    long row = baseRow + t;
                    var dt = Vector512.Load(delta + row * inner + c0);
                    var uu = Vector512.Load(u + row * inner + c0);
                    var du = dt * uu;
                    float* bc = xdbl + row * xw;
                    var acc = Vector512<float>.Zero;
                    for (int n = 0; n < states; n++)
                    {
                        var decay = VMath.Exp(dt * Vector512.Load(a + (long)n * inner + c0));
                        var hn = Vector512.FusedMultiplyAdd(decay, h[n], du * Vector512.Create(bc[bOff + n]));
                        h[n] = hn;
                        acc = Vector512.FusedMultiplyAdd(hn, Vector512.Create(bc[cOff + n]), acc);
                    }
                    var outV = Vector512.FusedMultiplyAdd(uu, dv, acc);
                    var z = Vector512.Load(xz + row * ldz + inner + c0);
                    Vector512.Store(outV * VMath.Silu(z), y + row * inner + c0);
                }
                if (hState != null)
                    for (int n = 0; n < states; n++) Vector512.Store(h[n], hState + (long)n * inner);
            }
            else
            {
                Span<float> h = stackalloc float[states * 16];
                for (int n = 0; n < states; n++)
                    for (int j = 0; j < 16; j++) h[n * 16 + j] = hState == null ? 0f : hState[(long)n * inner + j];
                for (int step = 0; step < length; step++)
                {
                    int t = reverse ? length - 1 - step : step;
                    long row = baseRow + t;
                    float* bc = xdbl + row * xw;
                    for (int j = 0; j < 16; j++)
                    {
                        int c = c0 + j;
                        float dt = delta[row * inner + c], uu = u[row * inner + c], du = dt * uu, acc = 0;
                        for (int n = 0; n < states; n++)
                        {
                            float hn = MathF.FusedMultiplyAdd(MathF.Exp(dt * a[(long)n * inner + c]), h[n * 16 + j], du * bc[bOff + n]);
                            h[n * 16 + j] = hn;
                            acc = MathF.FusedMultiplyAdd(hn, bc[cOff + n], acc);
                        }
                        y[row * inner + c] = MathF.FusedMultiplyAdd(uu, d[c], acc) * VMath.Silu(xz[row * ldz + inner + c]);
                    }
                }
                if (hState != null)
                    for (int n = 0; n < states; n++)
                        for (int j = 0; j < 16; j++) hState[(long)n * inner + j] = h[n * 16 + j];
            }
        });
    }

    public void Dispose()
    {
        InProj.Dispose();
        XProj.Dispose();
        DtProj.Dispose();
        OutProj.Dispose();
        ConvWeight.Dispose();
        ConvBias.Dispose();
        A.Dispose();
        D.Dispose();
    }
}

/// <summary>
/// Forward and backward scans projected together (<c>model.BidirectionalMamba</c>):
/// <c>LN(W_o · [f(x) + x, b(x) + x] + b_o)</c>; a causal one has only the forward scan, <c>LN(W_o · (f(x) + x) + b_o)</c>.
/// </summary>
internal sealed unsafe class BidirectionalMamba : IDisposable
{
    private const int TargetRows = 4096;

    public BidirectionalMamba(SafeTensorsFile f, string prefix, VoiceIsolatorConfig config, bool causal)
    {
        Width = config.Width;
        Causal = causal;
        ForwardBlocks = new MambaLayer(f, prefix + ".forward_blocks", config);
        if (!causal) BackwardBlocks = new MambaLayer(f, prefix + ".backward_blocks", config);
        OutputProj = Linear.Load(f, prefix + ".output_proj", Width, causal ? Width : 2 * Width);
        Norm = Norm.Load(f, prefix + ".norm", Width);
    }

    public int Width { get; }
    public bool Causal { get; }
    public MambaLayer ForwardBlocks { get; }
    public MambaLayer? BackwardBlocks { get; }
    public Linear OutputProj { get; }
    public Norm Norm { get; }

    /// <summary><c>x += this(x)</c> over <paramref name="x"/> <c>[sequences, length, width]</c> (the dual path's residual),
    /// a group of whole sequences at a time. <paramref name="state"/> carries a causal layer's state per sequence.</summary>
    public void AddTo(float* x, int sequences, int length, Workspace ws, Parallelism par, MambaState? state = null,
                      string traceName = "", TraceLayout layout = TraceLayout.BandMajor)
    {
        int W = Width, catW = Causal ? W : 2 * W;
        int group = ws.Trace != null ? sequences : Math.Clamp(TargetRows / Math.Max(1, length), 1, sequences);
        for (int s0 = 0; s0 < sequences; s0 += group)
        {
            int g = Math.Min(group, sequences - s0), rows = g * length;
            float* xg = x + (long)s0 * length * W;
            float* cat = ws.Get(Slot.Cat, (long)rows * catW);
            float* residual = ws.Get(Slot.Residual, (long)rows * W);
            par.ForRange(rows, 64, (r0, r1) =>
            {
                for (int r = r0; r < r1; r++)
                {
                    var src = new ReadOnlySpan<float>(xg + (long)r * W, W);
                    src.CopyTo(new Span<float>(cat + (long)r * catW, W));
                    if (!Causal) src.CopyTo(new Span<float>(cat + (long)r * catW + W, W));
                }
            });
            MambaState? sub = state == null ? null : state;
            if (state != null && (s0 != 0 || g != sequences))
                throw new InvalidOperationException("A carried state needs every sequence in one group.");
            ForwardBlocks.Forward(xg, g, length, reverse: false, cat, catW, accumulate: true, sub, ws, par);
            if (ws.Trace != null && traceName.Length > 0)
            {
                // the forward layer's own output: (x + f) - x
                float* f = ws.Get(Slot.Corrected, (long)rows * W);
                for (long r = 0; r < rows; r++)
                    TensorPrimitives.Subtract(new ReadOnlySpan<float>(cat + r * catW, W), new ReadOnlySpan<float>(xg + r * W, W),
                                              new Span<float>(f + r * W, W));
                ws.Emit(traceName + ".forward", f, (long)rows * W, layout);
            }
            BackwardBlocks?.Forward(xg, g, length, reverse: true, cat + W, catW, accumulate: true, null, ws, par);
            OutputProj.Forward(cat, catW, rows, residual, W, par);
            Norm.Forward(residual, residual, rows, par);
            if (traceName.Length > 0) ws.Emit(traceName, residual, (long)rows * W, layout);
            par.ForRange(rows, 256, (r0, r1) =>
            {
                long n = (long)(r1 - r0) * W;
                var dst = new Span<float>(xg + (long)r0 * W, (int)n);
                TensorPrimitives.Add(dst, new ReadOnlySpan<float>(residual + (long)r0 * W, (int)n), dst);
            });
        }
    }

    public void Dispose()
    {
        ForwardBlocks.Dispose();
        BackwardBlocks?.Dispose();
        OutputProj.Dispose();
        Norm.Dispose();
    }
}
