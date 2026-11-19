using System.Numerics.Tensors;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>
/// Streaming inference (the Python package's <c>streaming.Stream</c>): the isolated voice leaves as the audio arrives.
/// <para>
/// Samples are pushed in pieces of any size. Every block of up to <see cref="BlockFrames"/> new frames runs once with the
/// state carried (time attention over a cache of past keys rotated at their slots, the time Mamba's recurrence, the STFT's
/// overlap-add), and output samples are released as soon as every frame that touches them has been synthesized; the
/// latency is the STFT window plus the look-ahead frames. Only <see cref="Flush"/> ends a session.
/// </para>
/// <para>
/// With <c>windowFrames</c> and <c>crossfadeFrames</c> (600 and 200 by default, as the reference's <c>VoiceIsolator</c>
/// sets them) the time attention never looks back across a window boundary: a second lane starts
/// <c>crossfadeFrames</c> before each boundary with the Mamba states cloned and the attention empty, the decoder outputs
/// of the two lanes blend linearly over the crossfade, and the new lane carries on. The recurrence is kept throughout.
/// </para>
/// A session is not thread-safe; run sessions of one network concurrently from different threads if needed.
/// </summary>
public sealed unsafe class VoiceIsolatorStream : IDisposable
{
    private const int FirstSlots = 64;

    private readonly VoiceIsolatorNetwork _net;
    private readonly Workspace _ws;
    private readonly int _nFft, _hop, _bins, _width, _bands, _heads, _headDim;
    private readonly int _capacity, _attentionWindow;
    private object[] _states;
    private object[]? _statesB;
    private readonly List<(NativeBuffer<float> Keys, NativeBuffer<float> Values, int Slots)?> _spare = [];
    private readonly List<int> _boundaries = [];

    private float[] _audio = new float[1 << 16];
    private int _audioLength;
    private long _audioBase;
    private long _frames, _emitted, _pendingBase, _released, _olaBase;
    private readonly List<float[]> _raw = [];
    private readonly List<float[]> _spectra = [];
    private float[] _ola, _envelope;
    private bool _closed;

    /// <summary>Starts a session over <paramref name="network"/>, which must be causal (at a streaming position).</summary>
    public VoiceIsolatorStream(VoiceIsolatorNetwork network, int blockFrames = 4, int? cacheFrames = null, int? windowFrames = 600,
                               int? crossfadeFrames = 200)
    {
        if (network.Offline) throw new InvalidOperationException("Streaming needs a causal checkpoint (voice-isolator-v1-realtime).");
        if (blockFrames < 1) throw new ArgumentOutOfRangeException(nameof(blockFrames), "Process at least one frame per block.");
        if (windowFrames is 0) windowFrames = crossfadeFrames = null;
        if ((windowFrames is null) != (crossfadeFrames is null)) throw new ArgumentException("windowFrames and crossfadeFrames go together.");
        if (windowFrames != null && !(crossfadeFrames > 0 && crossfadeFrames < windowFrames))
            throw new ArgumentException("crossfadeFrames must be positive and shorter than windowFrames.");
        var config = network.Config;
        _net = network;
        _ws = network.RentWorkspace();
        BlockFrames = blockFrames;
        CacheFrames = cacheFrames ?? (config.AttentionWindowFrames > 0
            ? config.AttentionWindowFrames
            : Math.Max(1, (int)Math.Round(config.ChunkSeconds * config.SampleRate / config.HopLength, MidpointRounding.ToEven)));
        if (CacheFrames < 1) throw new ArgumentOutOfRangeException(nameof(cacheFrames));
        Lookahead = network.LookaheadFrames;
        WindowFrames = windowFrames;
        CrossfadeFrames = crossfadeFrames;
        _nFft = config.NFft;
        _hop = config.HopLength;
        _bins = config.Bins;
        _width = config.Width;
        _bands = network.Bands;
        _heads = config.NumHeads;
        _headDim = config.HeadDim;
        _attentionWindow = config.AttentionWindowFrames;
        int? lane = windowFrames + crossfadeFrames;
        _capacity = lane is int l && l <= CacheFrames + blockFrames ? l : CacheFrames + Math.Max(CacheFrames / 2, blockFrames);
        _states = NewStates();
        for (int i = 0; i < _states.Length; i++) _spare.Add(null);
        _ola = new float[_nFft];
        _envelope = new float[_nFft];
    }

    public int BlockFrames { get; }
    public int CacheFrames { get; }
    public int Lookahead { get; }
    public int? WindowFrames { get; }
    public int? CrossfadeFrames { get; }

    /// <summary>Frames at which a new attention lane took over (multiples of the window).</summary>
    public IReadOnlyList<int> Boundaries => _boundaries;

    /// <summary>Input samples that must follow an output sample before its release: the analysis window plus the look-ahead.</summary>
    public int LatencySamples => _nFft + _hop * Lookahead;

    public double LatencySeconds => (double)LatencySamples / _net.Config.SampleRate;

    // ---------------------------------------------------------------------------------------------------------------
    // Lane state
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>One lane's time-attention past for one block: keys (rotated at their slots) and values,
    /// <c>[rows, heads, slots, headDim]</c>, and the slots in use.</summary>
    private sealed class AttentionState
    {
        public NativeBuffer<float>? Keys, Values;
        public int Slots, End;
    }

    private object[] NewStates()
    {
        var c = _net.Config;
        var states = new object[_net.Blocks.Length];
        for (int i = 0; i < states.Length; i++)
            states[i] = _net.Blocks[i] is DualPathAttention ? new AttentionState() : new MambaState(_bands, c.MambaConv - 1, c.MambaState, c.MambaInner);
        return states;
    }

    private object[] CloneStates(object[] states)
    {
        var copy = new object[states.Length];
        for (int i = 0; i < states.Length; i++)
        {
            if (states[i] is MambaState m) copy[i] = m.Clone();
            else
            {
                var fresh = new AttentionState();
                if (_spare[i] is { } s)
                {
                    (fresh.Keys, fresh.Values, fresh.Slots) = (s.Keys, s.Values, s.Slots);
                    _spare[i] = null;
                }
                copy[i] = fresh;
            }
        }
        return copy;
    }

    private static void DisposeStates(object[]? states)
    {
        if (states == null) return;
        foreach (var s in states)
        {
            if (s is MambaState m) m.Dispose();
            else if (s is AttentionState a)
            {
                a.Keys?.Dispose();
                a.Values?.Dispose();
            }
        }
    }

    private void Reserve(AttentionState state, int slots)
    {
        if (slots <= state.Slots) return;
        int size = Math.Min(_capacity, Math.Max(slots, Math.Max(2 * state.Slots, FirstSlots)));
        long per = (long)size * _headDim;
        var keys = new NativeBuffer<float>((long)_bands * _heads * per, zero: false, hugePages: false);
        var values = new NativeBuffer<float>((long)_bands * _heads * per, zero: false, hugePages: false);
        if (state.End > 0)
        {
            long used = (long)state.End * _headDim * sizeof(float);
            for (int rh = 0; rh < _bands * _heads; rh++)
            {
                Buffer.MemoryCopy(state.Keys!.Ptr + (long)rh * state.Slots * _headDim, keys.Ptr + rh * per, used, used);
                Buffer.MemoryCopy(state.Values!.Ptr + (long)rh * state.Slots * _headDim, values.Ptr + rh * per, used, used);
            }
        }
        state.Keys?.Dispose();
        state.Values?.Dispose();
        (state.Keys, state.Values, state.Slots) = (keys, values, size);
    }

    /// <summary>Moves the last <see cref="CacheFrames"/> keys and values to the front, the keys rotated back by the
    /// distance they moved.</summary>
    private void Compact(AttentionState state)
    {
        int keep = Math.Min(state.End, CacheFrames), shift = state.End - keep;
        if (shift == 0) return;
        var (cos, sin) = Rope.Tables(_capacity, _headDim);
        int hd = _headDim;
        _net.Parallelism.For(_bands * _heads, rh =>
        {
            float* k = state.Keys!.Ptr + (long)rh * state.Slots * hd, v = state.Values!.Ptr + (long)rh * state.Slots * hd;
            for (int j = 0; j < keep; j++)
            {
                Buffer.MemoryCopy(k + (long)(shift + j) * hd, k + (long)j * hd, hd * 4L, hd * 4L);
                Rope.Apply(k + (long)j * hd, hd, cos, sin, shift, inverse: true);
                Buffer.MemoryCopy(v + (long)(shift + j) * hd, v + (long)j * hd, hd * 4L, hd * 4L);
            }
        });
        state.End = keep;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // One block of frames through the network
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>One attention block over a lane's new frames <c>[bands, frames, width]</c>, in place.</summary>
    private void Attend(DualPathAttention block, float* x, int frames, AttentionState state)
    {
        var time = block.Time;
        var par = _net.Parallelism;
        int W = _width, hd = _headDim, heads = _heads, rows = _bands * frames;
        long ld = 3L * W;
        float* h = _ws.Get(Slot.Norm, (long)rows * W);
        float* qkv = _ws.Get(Slot.Qkv, (long)rows * ld);
        time.Norm1.Forward(x, h, rows, par);
        time.Qkv.Forward(h, W, rows, qkv, ld, par);

        if (state.End + frames > _capacity) Compact(state);
        Reserve(state, state.End + frames);
        int first = state.End;
        var (cos, sin) = Rope.Tables(first + frames, hd);
        int slots = state.Slots;
        int past = Math.Min(first, CacheFrames);
        int window = _attentionWindow;
        bool exclusive = time.Exclusive;
        float* context = h;
        par.For(_bands * heads, rh =>
        {
            int band = rh / heads, head = rh % heads;
            float* keys = state.Keys!.Ptr + (long)rh * slots * hd, values = state.Values!.Ptr + (long)rh * slots * hd;
            for (int i = 0; i < frames; i++)
            {
                float* row = qkv + ((long)band * frames + i) * ld + head * hd;
                float* key = keys + (long)(first + i) * hd;
                Buffer.MemoryCopy(row + W, key, hd * 4L, hd * 4L);
                Rope.Apply(key, hd, cos, sin, first + i);
                Buffer.MemoryCopy(row + 2 * W, values + (long)(first + i) * hd, hd * 4L, hd * 4L);
            }
            float* scores = stackalloc float[past + frames];
            Span<float> query = stackalloc float[hd];
            fixed (float* q = query)
            {
                for (int i = 0; i < frames; i++)
                {
                    float* row = qkv + ((long)band * frames + i) * ld + head * hd;
                    Buffer.MemoryCopy(row, q, hd * 4L, hd * 4L);
                    Rope.Apply(q, hd, cos, sin, first + i);
                    int queryAt = first + i;
                    int lo = first - past, hi = queryAt + 1;
                    if (window > 0) lo = Math.Max(lo, queryAt - window + 1);
                    float* o = context + ((long)band * frames + i) * W + head * hd;
                    BandTransformer.AttendRow(q, keys + (long)lo * hd, values + (long)lo * hd, hd, hi - lo, hd, scores, o);
                    if (exclusive) TimeTransformer.ExclusiveProjection(o, row + 2 * W, hd);
                }
            }
        });
        state.End = first + frames;
        time.Proj.Forward(context, W, rows, x, W, par, accumulate: true);
        time.FeedForward(x, rows, _ws, par);
        float* t = _ws.Get(Slot.Transposed, (long)rows * W);
        Swap(x, t, _bands, frames, W);
        block.Band.Forward(t, frames, _bands, _ws, par);
        Swap(t, x, frames, _bands, W);
    }

    private static void Swap(float* src, float* dst, int a, int b, int width)
    {
        long bytes = width * sizeof(float);
        for (int i = 0; i < a; i++)
            for (int j = 0; j < b; j++)
                Buffer.MemoryCopy(src + ((long)i * b + j) * width, dst + ((long)j * a + i) * width, bytes, bytes);
    }

    /// <summary>The blocks over a lane's new frames <c>[bands, frames, width]</c>, in place.</summary>
    private void Backbone(float* x, int frames, object[] states)
    {
        for (int i = 0; i < _net.Blocks.Length; i++)
        {
            if (_net.Blocks[i] is DualPathAttention attention) Attend(attention, x, frames, (AttentionState)states[i]);
            else ((DualPathMamba)_net.Blocks[i]).Forward(x, _bands, frames, (MambaState)states[i], _ws, _net.Parallelism);
        }
    }

    /// <summary>The backbone and decoder over the spectra of <paramref name="count"/> new frames; returns the raw decoder
    /// output <c>[count, bins, 4]</c> (blended across lanes inside a crossfade).</summary>
    private float* RunBlock(float* spectra, int count)
    {
        var par = _net.Parallelism;
        long features = (long)_bands * count * _width;
        float* x = _ws.Get(Slot.Features, features);
        _net.Codec.Encode(spectra, 2L * _bins, count, x, _ws, par);
        float* raw = _ws.Get(Slot.Raw, 4L * _bins * count);
        if (WindowFrames is not int windowFrames)
        {
            Backbone(x, count, _states);
            _net.Codec.Decode(x, count, raw, 4L * _bins, _ws, par);
            return raw;
        }
        int crossfadeFrames = CrossfadeFrames!.Value;
        int first = (int)(_frames % windowFrames), crossfadeStart = windowFrames - crossfadeFrames;
        if (first + count > windowFrames || (first < crossfadeStart && crossfadeStart < first + count))
            throw new InvalidOperationException("A block crossed a crossfade start or a window boundary.");
        if (first >= crossfadeStart && _statesB == null) _statesB = CloneStates(_states);
        if (first < crossfadeStart)
        {
            Backbone(x, count, _states);
            _net.Codec.Decode(x, count, raw, 4L * _bins, _ws, par);
            return raw;
        }
        float* xb = _ws.Get(Slot.LaneB, features);
        Buffer.MemoryCopy(x, xb, features * 4, features * 4);
        Backbone(x, count, _states);
        _net.Codec.Decode(x, count, raw, 4L * _bins, _ws, par);
        Backbone(xb, count, _statesB!);
        float* rawB = _ws.Get(Slot.RawB, 4L * _bins * count);
        _net.Codec.Decode(xb, count, rawB, 4L * _bins, _ws, par);
        int n = 4 * _bins;
        for (int k = 0; k < count; k++)
        {
            float weight = Math.Clamp((float)(first + k - crossfadeStart + 1) / crossfadeFrames, 0f, 1f);
            float* a = raw + (long)k * n, b = rawB + (long)k * n;
            for (int j = 0; j < n; j++) a[j] = a[j] * (1f - weight) + b[j] * weight;
        }
        if (first + count == windowFrames)
        {
            _boundaries.Add((int)(_frames + count));
            for (int i = 0; i < _states.Length; i++)
            {
                if (_states[i] is AttentionState a && a.Keys != null)
                {
                    _spare[i] = (a.Keys, a.Values!, a.Slots);
                    a.Keys = a.Values = null;
                }
            }
            DisposeStates(_states);
            _states = _statesB!;
            _statesB = null;
        }
        return raw;
    }

    private long NextCut()
    {
        if (WindowFrames is not int w) return _frames + BlockFrames;
        long start = _frames - _frames % w;
        long crossfadeStart = start + w - CrossfadeFrames!.Value;
        return _frames < crossfadeStart ? crossfadeStart : start + w;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Analysis, synthesis and release
    // ---------------------------------------------------------------------------------------------------------------

    private long Total => _audioBase + _audioLength;

    /// <summary>Spectra of frames [first, first + count): each frame's window of the audio, zeros where there is none.</summary>
    private void Analyze(long first, int count, float* spectra)
    {
        int n = _nFft;
        long total = Total;
        var spectral = _net.Spectral;
        _net.Parallelism.ForRange(count, 2, (k0, k1) =>
        {
            Span<float> segment = stackalloc float[n];
            for (int k = k0; k < k1; k++)
            {
                long start = (first + k) * _hop - n / 2;
                segment.Clear();
                long lo = Math.Max(start, _audioBase), hi = Math.Min(start + n, total);
                if (hi > lo) _audio.AsSpan((int)(lo - _audioBase), (int)(hi - lo)).CopyTo(segment[(int)(lo - start)..]);
                spectral.AnalyzeFrame(segment, new Span<float>(spectra + 2L * _bins * k, 2 * _bins));
            }
        });
    }

    /// <summary>Computes every frame below <paramref name="limit"/> in blocks, then synthesizes what is ready.</summary>
    private void Advance(long limit)
    {
        while (_frames < limit)
        {
            int count = (int)Math.Min(BlockFrames, Math.Min(limit - _frames, NextCut() - _frames));
            float* spectra = _ws.Get(Slot.BlockSpectra, 2L * _bins * count);
            Analyze(_frames, count, spectra);
            float* raw = RunBlock(spectra, count);
            for (int k = 0; k < count; k++)
            {
                _raw.Add(new ReadOnlySpan<float>(raw + 4L * _bins * k, 4 * _bins).ToArray());
                _spectra.Add(new ReadOnlySpan<float>(spectra + 2L * _bins * k, 2 * _bins).ToArray());
            }
            _frames += count;
        }
        while (_emitted + Lookahead < _frames) Synthesize();
        Trim();
    }

    /// <summary>Synthesizes the next frame from the decoder's output <see cref="Lookahead"/> frames later; the last
    /// computed frame serves every frame beyond it.</summary>
    private void Synthesize()
    {
        long index = _emitted, last = _frames - 1;
        long source = index <= last ? Math.Min(index + Lookahead, last) : index;
        var raw = _raw[(int)(source - _pendingBase)];
        var spectrum = _spectra[(int)(index - _pendingBase)];
        Span<float> corrected = stackalloc float[2 * _bins];
        Span<float> frame = stackalloc float[_nFft];
        fixed (float* r = raw, s = spectrum, c = corrected)
            _net.Spectral.Correction(r, 4L * _bins, s, 2L * _bins, 1, c, 2L * _bins, Parallelism.Single);
        _net.Spectral.SynthesizeFrame(corrected, frame);
        int start = (int)(index * _hop - _olaBase);
        int needed = start + _nFft;
        if (needed > _ola.Length)
        {
            int size = Math.Max(needed, 2 * _ola.Length);
            Array.Resize(ref _ola, size);
            Array.Resize(ref _envelope, size);
        }
        var ola = _ola.AsSpan(start, _nFft);
        TensorPrimitives.Add(ola, frame, ola);
        var env = _envelope.AsSpan(start, _nFft);
        TensorPrimitives.Add(env, _net.Spectral.WindowSquared, env);
        _emitted++;
    }

    /// <summary>Output samples [released, upto): the input plus the synthesized change.</summary>
    private float[] Release(long upto)
    {
        long start = _released, stop = Math.Min(upto, Total);
        if (stop <= start) return [];
        var output = new float[stop - start];
        long offset = _nFft / 2 - _olaBase;
        for (long s = start; s < stop; s++)
        {
            long at = s + offset;
            float delta = at < _ola.Length ? _ola[at] / MathF.Max(_envelope[at], 1e-11f) : 0f;
            output[s - start] = _audio[s - _audioBase] + delta;
        }
        _released = stop;
        return output;
    }

    /// <summary>Drops what no frame still needs.</summary>
    private void Trim()
    {
        int drop = (int)(_emitted - _pendingBase);
        if (drop > 0)
        {
            _raw.RemoveRange(0, drop);
            _spectra.RemoveRange(0, drop);
            _pendingBase = _emitted;
        }
        long keepFrom = _released - _olaBase;
        if (keepFrom > 0)
        {
            int kept = (int)Math.Max(0, _ola.Length - keepFrom);
            Array.Copy(_ola, keepFrom, _ola, 0, kept);
            Array.Copy(_envelope, keepFrom, _envelope, 0, kept);
            Array.Clear(_ola, kept, _ola.Length - kept);
            Array.Clear(_envelope, kept, _envelope.Length - kept);
            _olaBase = _released;
        }
        long neededFrom = Math.Min(_released, Math.Max(0, _frames * _hop - _nFft / 2));
        long dropAudio = neededFrom - _audioBase;
        if (dropAudio > 0)
        {
            Array.Copy(_audio, dropAudio, _audio, 0, _audioLength - dropAudio);
            _audioLength -= (int)dropAudio;
            _audioBase = neededFrom;
        }
    }

    private static long FloorDiv(long a, long b) => a >= 0 ? a / b : -((-a + b - 1) / b);

    /// <summary>Feeds new mono samples at the model rate; returns the output samples that became final.</summary>
    public float[] Push(ReadOnlySpan<float> samples)
    {
        ObjectDisposedException.ThrowIf(_states.Length == 0, this);
        if (_closed) throw new InvalidOperationException("The stream was flushed.");
        if (_audioLength + samples.Length > _audio.Length) Array.Resize(ref _audio, Math.Max(_audioLength + samples.Length, 2 * _audio.Length));
        samples.CopyTo(_audio.AsSpan(_audioLength));
        _audioLength += samples.Length;
        long computable = FloorDiv(Total - _nFft / 2, _hop) + 1;
        Advance(Math.Max(0, computable));
        var released = Release(_emitted * _hop - _nFft / 2);
        Trim();
        return released;
    }

    /// <summary>Ends the stream and returns every remaining output sample.</summary>
    public float[] Flush()
    {
        if (_closed) return [];
        _closed = true;
        long total = Total;
        Advance(total / _hop + 1);
        while (_emitted < _frames) Synthesize();
        return Release(total);
    }

    public void Dispose()
    {
        DisposeStates(_states);
        DisposeStates(_statesB);
        foreach (var s in _spare)
        {
            s?.Keys.Dispose();
            s?.Values.Dispose();
        }
        _states = [];
        _statesB = null;
        _net.ReturnWorkspace(_ws);
    }
}
