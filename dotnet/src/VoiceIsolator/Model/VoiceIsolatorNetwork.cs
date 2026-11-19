using VoiceIsolation.IO;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>A block of the backbone over band tokens <c>[bands, frames, width]</c>.</summary>
internal abstract unsafe class Block : IDisposable
{
    /// <summary>Runs the block in place over a whole clip's tokens.</summary>
    public abstract void Forward(float* x, int bands, int frames, bool causal, Workspace ws, Parallelism par);

    public abstract void Dispose();

    /// <summary><c>[a, b, width]</c> to <c>[b, a, width]</c>, a token (a row of width floats) at a time.</summary>
    protected static void Swap(float* src, float* dst, int a, int b, int width, Parallelism par)
    {
        long bytes = width * sizeof(float);
        par.ForRange(a, 4, (i0, i1) =>
        {
            for (int i = i0; i < i1; i++)
                for (int j = 0; j < b; j++)
                    Buffer.MemoryCopy(src + ((long)i * b + j) * width, dst + ((long)j * a + i) * width, bytes, bytes);
        });
    }
}

/// <summary>Attention along time within each band, then across the bands of each frame (<c>DualPathAttention</c>).</summary>
internal sealed unsafe class DualPathAttention : Block
{
    public DualPathAttention(SafeTensorsFile f, string prefix, VoiceIsolatorConfig config)
    {
        Time = new TimeTransformer(f, prefix + ".time_transformer", config, config.ExclusiveOnTime);
        Band = new BandTransformer(f, prefix + ".band_transformer", config, config.ExclusiveOnBand);
    }

    public TimeTransformer Time { get; }
    public BandTransformer Band { get; }

    public override void Forward(float* x, int bands, int frames, bool causal, Workspace ws, Parallelism par)
    {
        int W = Time.Width;
        Time.Forward(x, bands, frames, causal, ws, par);
        ws.Emit("time", x, (long)bands * frames * W);
        float* t = ws.Get(Slot.Transposed, (long)bands * frames * W);
        Swap(x, t, bands, frames, W, par);
        Band.Forward(t, frames, bands, ws, par);
        Swap(t, x, frames, bands, W, par);
    }

    public override void Dispose()
    {
        Time.Dispose();
        Band.Dispose();
    }
}

/// <summary>Mamba along time within each band, then across the bands of each frame (<c>DualPathMamba</c>).</summary>
internal sealed unsafe class DualPathMamba : Block
{
    public DualPathMamba(SafeTensorsFile f, string prefix, VoiceIsolatorConfig config)
    {
        TimeMamba = new BidirectionalMamba(f, prefix + ".time_mamba", config, config.Causal);
        FreqMamba = new BidirectionalMamba(f, prefix + ".freq_mamba", config, causal: false);
    }

    public BidirectionalMamba TimeMamba { get; }
    public BidirectionalMamba FreqMamba { get; }

    public override void Forward(float* x, int bands, int frames, bool causal, Workspace ws, Parallelism par) =>
        Forward(x, bands, frames, null, ws, par);

    /// <summary>The block, with the time Mamba's state carried per band when <paramref name="state"/> is given.</summary>
    public void Forward(float* x, int bands, int frames, MambaState? state, Workspace ws, Parallelism par)
    {
        int W = TimeMamba.Width;
        TimeMamba.AddTo(x, bands, frames, ws, par, state, "time", TraceLayout.BandMajor);
        float* t = ws.Get(Slot.Transposed, (long)bands * frames * W);
        Swap(x, t, bands, frames, W, par);
        FreqMamba.AddTo(t, frames, bands, ws, par, null, "freq", TraceLayout.FrameMajor);
        Swap(t, x, frames, bands, W, par);
    }

    public override void Dispose()
    {
        TimeMamba.Dispose();
        FreqMamba.Dispose();
    }
}

/// <summary>
/// The isolator's network (<c>VoiceIsolatorNetwork</c>): STFT, band encoders, the dual-path blocks, the per-band decoder
/// heads, the bounded correction and the iSTFT, over one chunk of 48 kHz mono audio. <see cref="Enhance"/> renders clips
/// of any length in overlapping chunks, as the reference does. A causal checkpoint runs at one of its trained look-ahead
/// positions (<see cref="SetLookahead"/>); the decoder's output for frame <c>t</c> is then read from frame
/// <c>t + lookahead</c>. Instances are thread-safe: every call has its own workspace.
/// </summary>
public sealed unsafe class VoiceIsolatorNetwork : IDisposable
{
    private readonly Lock _workspaceGate = new();
    private readonly Stack<Workspace> _workspaces = new();
    private int _lookaheadIndex = -1;

    /// <summary>Builds the network from a safetensors checkpoint carrying its architecture in the <c>config</c>
    /// metadata. Weights are packed into the GEMM layout; the file can be disposed afterwards.</summary>
    public VoiceIsolatorNetwork(SafeTensorsFile file, int threads = 0)
    {
        if (!file.Metadata.TryGetValue("config", out var json))
            throw new InvalidDataException($"{file.Source}: safetensors metadata lacks the 'config' entry.");
        Config = VoiceIsolatorConfig.Parse(json);
        Metadata = file.Metadata;
        Edges = Config.BandEdges();
        Parallelism = new Parallelism(threads);
        Spectral = new Spectral(Config.NFft, Config.HopLength, Config.CorrectionBound);
        CheckKeys(file);
        Codec = new BandCodec(file, Config, Edges);
        Blocks = new Block[Config.BlockTypes.Length];
        for (int i = 0; i < Blocks.Length; i++)
            Blocks[i] = Config.BlockTypes[i] == "transformer"
                ? new DualPathAttention(file, $"blocks.{i}", Config)
                : new DualPathMamba(file, $"blocks.{i}", Config);
        if (Config.LookaheadFramesChoices.Length > 0) SetLookahead(Config.LookaheadFrames);
    }

    /// <summary>Loads a <c>.safetensors</c> checkpoint or a packed <c>.vimodel</c>.</summary>
    public static VoiceIsolatorNetwork Load(string path, int threads = 0, string? expectedSha256 = null)
    {
        Span<byte> head = stackalloc byte[8];
        using (var probe = File.OpenRead(path)) probe.ReadAtLeast(head, 8, throwOnEndOfStream: false);
        if (ModelArchive.IsArchive(head))
        {
            using var stream = File.OpenRead(path);
            return Load(stream, threads, expectedSha256, path);
        }
        if (expectedSha256 != null) VerifySha256(path, expectedSha256);
        using var file = SafeTensorsFile.Open(path);
        return new VoiceIsolatorNetwork(file, threads);
    }

    /// <summary>Loads a packed <c>.vimodel</c> from a stream (an embedded resource, for example).</summary>
    public static VoiceIsolatorNetwork Load(Stream packed, int threads = 0, string? expectedSha256 = null, string name = "model")
    {
        using var file = SafeTensorsFile.FromBuffer(ModelArchive.Unpack(packed, expectedSha256, name), name);
        return new VoiceIsolatorNetwork(file, threads);
    }

    private static void VerifySha256(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        string actual = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(stream));
        if (!actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Checksum mismatch for {path}: expected {expected}, got {actual}.");
    }

    public VoiceIsolatorConfig Config { get; }

    /// <summary>The checkpoint's <c>__metadata__</c> (format, num_params, voice_policy, config).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    public (int Start, int End)[] Edges { get; }
    public int Bands => Edges.Length;
    internal Parallelism Parallelism { get; }
    internal Spectral Spectral { get; }
    internal BandCodec Codec { get; }
    internal Block[] Blocks { get; }

    /// <summary>Whether every frame sees its whole chunk (a two-sided checkpoint).</summary>
    public bool Offline => !Config.Causal;

    /// <summary>The look-ahead in force, in frames; zero offline.</summary>
    public int LookaheadFrames => Offline ? 0 : _lookaheadIndex >= 0 ? Config.LookaheadFramesChoices[_lookaheadIndex] : Config.LookaheadFrames;

    /// <summary>Look-ahead positions, in frames, the checkpoint was trained at (empty for an offline one).</summary>
    public IReadOnlyList<int> LookaheadChoices =>
        Config.LookaheadFramesChoices.Length > 0 ? Config.LookaheadFramesChoices : Config.Causal ? [Config.LookaheadFrames] : [];

    /// <summary>Moves a causal checkpoint to the trained streaming position with this look-ahead.</summary>
    public void SetLookahead(int frames)
    {
        var choices = Config.LookaheadFramesChoices;
        if (choices.Length == 0)
        {
            if (frames != Config.LookaheadFrames || !Config.Causal)
                throw new ArgumentException("This checkpoint has no streaming positions to choose from.", nameof(frames));
            return;
        }
        int index = Array.IndexOf(choices, frames);
        if (index < 0) throw new ArgumentException($"Look-ahead {frames} is not among the trained positions [{string.Join(", ", choices)}].", nameof(frames));
        lock (_workspaceGate)
        {
            _lookaheadIndex = index;
            Codec.SetLookahead(index);
        }
    }

    /// <summary>Learned parameters, from the checkpoint's metadata when present.</summary>
    public long ParameterCount => Metadata.TryGetValue("num_params", out var n) && long.TryParse(n, out long v) ? v : 0;

    private void CheckKeys(SafeTensorsFile file)
    {
        string[] trainingOnly = ["discriminator.", "disc.", "ema.", "optimizer.", "scheduler.", "enhance_decoder.", "decision_head."];
        var spare = file.Tensors.Keys.Where(k => !k.StartsWith("blocks.") && !k.StartsWith("encoders.") && !k.StartsWith("denoise_decoder.")
                                                 && k is not ("band_embedding" or "lookahead_embedding.weight")
                                                 && !trainingOnly.Any(k.StartsWith)).ToList();
        if (spare.Count > 0)
            throw new InvalidDataException($"{file.Source}: checkpoint does not match the architecture: unexpected keys [{string.Join(", ", spare.Take(4))}].");
    }

    internal Workspace RentWorkspace()
    {
        lock (_workspaceGate) return _workspaces.Count > 0 ? _workspaces.Pop() : new Workspace();
    }

    internal void ReturnWorkspace(Workspace ws)
    {
        ws.Trace = null;
        ws.TracePrefix = "";
        lock (_workspaceGate)
        {
            if (_workspaces.Count < Environment.ProcessorCount) _workspaces.Push(ws);
            else ws.Dispose();
        }
    }

    /// <summary>
    /// Isolates one chunk of mono audio at the model rate in a single pass (<c>VoiceIsolatorNetwork.forward</c>): every
    /// frame sees the whole chunk offline, or its past (and the look-ahead) for a causal checkpoint.
    /// <paramref name="trace"/> receives every stage's output for validation: <c>stft</c>, <c>encode</c>,
    /// <c>block{i}.time</c> (and <c>.time.attn</c>, <c>.time.forward</c>, <c>.freq</c>), <c>block{i}</c>, <c>decoder</c>,
    /// <c>correction</c> and <c>output</c>.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> audio, TraceCallback? trace = null)
    {
        if (audio.IsEmpty) throw new ArgumentException("Expected nonempty mono audio.", nameof(audio));
        var ws = RentWorkspace();
        try
        {
            ws.Trace = trace;
            return Forward(audio, ws);
        }
        finally
        {
            ReturnWorkspace(ws);
        }
    }

    private float[] Forward(ReadOnlySpan<float> audio, Workspace ws)
    {
        var par = Parallelism;
        int W = Config.Width, bins = Config.Bins, bands = Bands;
        int frames = Spectral.Frames(audio.Length);
        long ldSpec = 2L * bins, ldRaw = 4L * bins;
        float* spectrum = ws.Get(Slot.Spectrum, frames * ldSpec);
        Profiler.Start();
        Spectral.Stft(audio, spectrum, par);
        Profiler.Mark("stft");
        ws.Emit("stft", spectrum, frames * ldSpec, TraceLayout.Other);

        float* features = ws.Get(Slot.Features, (long)bands * frames * W);
        Codec.Encode(spectrum, ldSpec, frames, features, ws, par);
        Profiler.Mark("encode");
        ws.Emit("encode", features, (long)bands * frames * W);
        RunBlocks(features, frames, ws, par);

        float* raw = ws.Get(Slot.Raw, frames * ldRaw);
        Codec.Decode(features, frames, raw, ldRaw, ws, par);
        Profiler.Mark("decode");
        ws.Emit("decoder", raw, frames * ldRaw, TraceLayout.Other);
        int lookahead = LookaheadFrames;
        float* corrected = ws.Get(Slot.Corrected, frames * ldSpec);
        if (lookahead > 0)
        {
            // read_ahead: frame t takes the decoder's output of frame min(t + lookahead, last)
            par.ForRange(frames, 16, (f0, f1) =>
            {
                for (int t = f0; t < f1; t++)
                {
                    int source = Math.Min(t + lookahead, frames - 1);
                    Spectral.Correction(raw + source * ldRaw, ldRaw, spectrum + t * ldSpec, ldSpec, 1, corrected + t * ldSpec, ldSpec,
                                        Parallelism.Single);
                }
            });
        }
        else
        {
            Spectral.Correction(raw, ldRaw, spectrum, ldSpec, frames, corrected, ldSpec, par);
        }
        ws.Emit("correction", corrected, frames * ldSpec, TraceLayout.Other);
        float[] output = Spectral.Istft(corrected, ldSpec, frames, audio.Length, par);
        System.Numerics.Tensors.TensorPrimitives.Add(output, audio, output);
        Profiler.Mark("correction + istft");
        fixed (float* o = output) ws.Emit("output", o, output.Length, TraceLayout.Other);
        return output;
    }

    private void RunBlocks(float* features, int frames, Workspace ws, Parallelism par)
    {
        bool causal = !Offline;
        int W = Config.Width;
        for (int i = 0; i < Blocks.Length; i++)
        {
            ws.TracePrefix = $"block{i}.";
            Blocks[i].Forward(features, Bands, frames, causal, ws, par);
            Profiler.Mark(Blocks[i] is DualPathAttention ? "attention block" : "mamba block");
            ws.TracePrefix = "";
            ws.Emit($"block{i}", features, (long)Bands * frames * W);
        }
    }

    /// <summary>The render window and its crossfade in samples for the given seconds (<c>model.window_samples</c>).</summary>
    public (int Chunk, int Overlap) WindowSamples(double? chunkSeconds = null, double? overlapSeconds = null)
    {
        double chunkS = chunkSeconds ?? Config.ChunkSeconds, overlapS = overlapSeconds ?? Config.OverlapSeconds;
        if (!double.IsFinite(chunkS) || !double.IsFinite(overlapS)) throw new ArgumentException("Window lengths must be finite.");
        int chunk = (int)(chunkS * Config.SampleRate);
        if (chunk < Config.HopLength) throw new ArgumentException($"chunk_seconds={chunkS} is shorter than {1000.0 * Config.HopLength / Config.SampleRate:G} ms.");
        if (overlapS < 0) throw new ArgumentException($"overlap_seconds must not be negative, not {overlapS}.");
        return (chunk, Math.Min((int)(overlapS * Config.SampleRate), chunk / 2));
    }

    /// <summary>
    /// Isolates mono audio of any length at the model rate (<c>VoiceIsolatorNetwork.enhance</c>). Windows of
    /// <paramref name="chunkSeconds"/> (the checkpoint's own by default) overlap by <paramref name="overlapSeconds"/> with a
    /// linear crossfade; a clip no longer than one window is a single pass. Every window of a longer clip is a full one: the
    /// last ends at the clip's end and fades in where the window before it fades out.
    /// </summary>
    public float[] Enhance(ReadOnlySpan<float> audio, double? chunkSeconds = null, double? overlapSeconds = null,
                           Action<int, int>? progress = null)
    {
        var (chunk, overlap) = WindowSamples(chunkSeconds, overlapSeconds);
        int n = audio.Length;
        if (n <= chunk)
        {
            var single = Forward(audio);
            progress?.Invoke(1, 1);
            return single;
        }
        var starts = new List<int>();
        for (int s = 0; s < n - chunk; s += chunk - overlap) starts.Add(s);
        starts.Add(n - chunk);
        var output = new float[n];
        var weight = new float[n];
        var w = new float[chunk];
        for (int index = 0; index < starts.Count; index++)
        {
            int start = starts[index], end = start + chunk;
            w.AsSpan().Fill(1f);
            if (index > 0)
            {
                int fadeIn = starts[index - 1] + chunk - overlap - start;
                w.AsSpan(0, fadeIn).Clear();
                if (overlap > 0) Linspace(w.AsSpan(fadeIn, overlap), 0f, 1f);
            }
            if (overlap > 0 && end < n) Linspace(w.AsSpan(chunk - overlap, overlap), 1f, 0f);
            var y = Forward(audio.Slice(start, chunk));
            for (int i = 0; i < chunk; i++)
            {
                output[start + i] += y[i] * w[i];
                weight[start + i] += w[i];
            }
            progress?.Invoke(index + 1, starts.Count);
        }
        for (int i = 0; i < n; i++) output[i] /= MathF.Max(weight[i], 1e-6f);
        return output;
    }

    /// <summary><c>torch.linspace(start, end, count)</c> in float32.</summary>
    internal static void Linspace(Span<float> dst, float start, float end)
    {
        int count = dst.Length;
        if (count == 1)
        {
            dst[0] = start;
            return;
        }
        float step = (end - start) / (count - 1);
        int half = count / 2;
        for (int i = 0; i < count; i++) dst[i] = i < half ? start + step * i : end - step * (count - 1 - i);
    }

    public void Dispose()
    {
        foreach (var b in Blocks) b.Dispose();
        Codec.Dispose();
        Parallelism.Dispose();
        lock (_workspaceGate)
            while (_workspaces.Count > 0) _workspaces.Pop().Dispose();
    }
}
