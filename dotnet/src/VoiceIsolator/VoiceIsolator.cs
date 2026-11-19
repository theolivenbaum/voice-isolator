using VoiceIsolation.Audio;
using VoiceIsolation.IO;
using VoiceIsolation.Model;

namespace VoiceIsolation;

/// <summary>How a checkpoint runs: offline (every frame sees its whole chunk) or as a causal stream.</summary>
public enum IsolationMode
{
    Offline,
    Streaming,
}

/// <summary>Settings of a <see cref="VoiceIsolator"/>; the defaults are the Python package's.</summary>
public sealed record VoiceIsolatorOptions
{
    /// <summary>Worker threads (0: all cores; 1: the calling thread only).</summary>
    public int Threads { get; init; }

    /// <summary>Streaming look-ahead in frames (10 ms each), one of the checkpoint's trained positions (0, 2 or 4 for the
    /// released realtime model); null takes the checkpoint's own (2 frames, 20 ms).</summary>
    public int? LookaheadFrames { get; init; }

    /// <summary>Frames computed together in a stream: a throughput knob, not a latency one.</summary>
    public int BlockFrames { get; init; } = 4;

    /// <summary>How far back the stream's time attention may look, in frames; null takes the checkpoint's attention
    /// window (else its training chunk).</summary>
    public int? CacheFrames { get; init; }

    /// <summary>Attention windows of the stream (frames); 0 turns them off, the attention then looking back over its whole
    /// cache. With <see cref="CrossfadeFrames"/>, keeps a long session from drifting.</summary>
    public int WindowFrames { get; init; } = 600;

    /// <summary>Frames over which a new attention lane fades in before each window boundary.</summary>
    public int CrossfadeFrames { get; init; } = 200;

    /// <summary>Offline render window in seconds; null takes the checkpoint's own (8 s for the released model).</summary>
    public double? ChunkSeconds { get; init; }

    /// <summary>Offline crossfade between windows in seconds; null takes the checkpoint's own (2 s).</summary>
    public double? OverlapSeconds { get; init; }

    /// <summary>Offline passes (the model run again over its own output); null takes the checkpoint's own (1).</summary>
    public int? OfflinePasses { get; init; }
}

/// <summary>
/// Isolates voices at 48 kHz (the Python package's <c>VoiceIsolator</c>): <see cref="VoiceIsolatorModel.Offline"/> for
/// recordings at the best quality, <see cref="VoiceIsolatorModel.Realtime"/> for live audio as a causal stream.
/// <code>
/// using var isolator = await VoiceIsolator.LoadAsync();                // voice-isolator-v1
/// var (wav, rate) = WaveFile.Read("noisy.wav");
/// float[] voice = isolator.Isolate(wav, rate);                         // at the input rate; wav - voice is the background
/// </code>
/// One instance is thread-safe and can serve concurrent <see cref="Isolate"/> calls; streams are per session.
/// </summary>
public sealed class VoiceIsolator : IDisposable
{
    /// <summary>Seconds of audio pushed at a time when <see cref="Isolate"/> runs the stream over a clip.</summary>
    public const double StreamPieceSeconds = 0.5;

    private VoiceIsolator(VoiceIsolatorNetwork network, VoiceIsolatorOptions? options)
    {
        Network = network;
        Options = options ?? new VoiceIsolatorOptions();
        Mode = network.Offline ? IsolationMode.Offline : IsolationMode.Streaming;
        if (Mode == IsolationMode.Streaming)
        {
            LookaheadFrames = Options.LookaheadFrames ?? network.LookaheadFrames;
            if (Options.OfflinePasses > 1) throw new ArgumentException("OfflinePasses above 1 applies to an offline checkpoint.");
        }
        OfflinePasses = Options.OfflinePasses ?? network.Config.OfflinePasses;
        if (OfflinePasses < 1) throw new ArgumentException("OfflinePasses must be at least 1.");
        if (Options.BlockFrames < 1) throw new ArgumentException("BlockFrames must be at least 1.");
    }

    public VoiceIsolatorNetwork Network { get; }
    public VoiceIsolatorOptions Options { get; }
    public IsolationMode Mode { get; }
    public int OfflinePasses { get; }

    /// <summary>The sample rate the model runs at, in Hz (48000).</summary>
    public int SampleRate => Network.Config.SampleRate;

    /// <summary>The duration of one model frame, the unit of the look-ahead.</summary>
    public double FrameSeconds => (double)Network.Config.HopLength / SampleRate;

    /// <summary>The look-ahead of the streaming mode, in frames (one of <see cref="LookaheadChoices"/>).</summary>
    public int LookaheadFrames
    {
        get => Network.LookaheadFrames;
        set
        {
            if (Mode != IsolationMode.Streaming)
            {
                if (value != 0) throw new InvalidOperationException("An offline checkpoint has no look-ahead.");
                return;
            }
            Network.SetLookahead(value);
        }
    }

    /// <summary>The look-ahead positions, in frames, the checkpoint was trained at.</summary>
    public IReadOnlyList<int> LookaheadChoices => Network.LookaheadChoices;

    /// <summary>The streaming algorithmic latency in samples: the analysis window plus the look-ahead.</summary>
    public int LatencySamples => Network.Config.NFft + Network.Config.HopLength * LookaheadFrames;

    /// <summary>Wraps a loaded network.</summary>
    public static VoiceIsolator Create(VoiceIsolatorNetwork network, VoiceIsolatorOptions? options = null) => new(network, options);

    /// <summary>Loads a local <c>.safetensors</c> checkpoint or packed <c>.vimodel</c>.</summary>
    public static VoiceIsolator Load(string path, VoiceIsolatorOptions? options = null) =>
        new(VoiceIsolatorNetwork.Load(path, options?.Threads ?? 0), options);

    /// <summary>
    /// Loads a released model: from its model package when referenced (<c>VoiceIsolatorSharp.Models.Offline</c> /
    /// <c>.Realtime</c>), else from the cache, else downloaded once from the pinned Hub revision. Every source is checked
    /// against the release checksum.
    /// </summary>
    public static async Task<VoiceIsolator> LoadAsync(VoiceIsolatorModel model = VoiceIsolatorModel.Offline, VoiceIsolatorOptions? options = null,
                                                      string? downloadToPath = null, Action<DownloadProgress>? reportProgress = null,
                                                      CancellationToken cancellationToken = default)
    {
        int threads = options?.Threads ?? 0;
        string sha = VoiceIsolatorModels.GetSha256(model);
        await using (var embedded = VoiceIsolatorModels.TryOpenEmbedded(model))
        {
            if (embedded != null)
                return new VoiceIsolator(VoiceIsolatorNetwork.Load(embedded, threads, sha, VoiceIsolatorModels.GetResourceName(model)), options);
        }
        string path = await DownloadModelAsync(model, downloadToPath, reportProgress, cancellationToken).ConfigureAwait(false);
        try
        {
            return new VoiceIsolator(VoiceIsolatorNetwork.Load(path, threads, sha), options);
        }
        catch (InvalidDataException) when (downloadToPath == null)
        {
            // A corrupt cached file: fetch it again once.
            File.Delete(path);
            path = await DownloadModelAsync(model, null, reportProgress, cancellationToken).ConfigureAwait(false);
            return new VoiceIsolator(VoiceIsolatorNetwork.Load(path, threads, sha), options);
        }
    }

    /// <summary>Loads a local checkpoint path, or a released model by name (see <see cref="VoiceIsolatorModels.TryParse"/>).</summary>
    public static Task<VoiceIsolator> LoadAsync(string modelOrPath, VoiceIsolatorOptions? options = null,
                                                Action<DownloadProgress>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (File.Exists(modelOrPath)) return Task.FromResult(Load(modelOrPath, options));
        if (modelOrPath == VoiceIsolatorModels.Repository) return LoadAsync(VoiceIsolatorModel.Offline, options, null, reportProgress, cancellationToken);
        if (VoiceIsolatorModels.TryParse(modelOrPath, out var model)) return LoadAsync(model, options, null, reportProgress, cancellationToken);
        throw new ArgumentException($"'{modelOrPath}' is neither a local file nor a known model (voice-isolator-v1, voice-isolator-v1-realtime).",
                                    nameof(modelOrPath));
    }

    /// <summary>Downloads a released checkpoint without loading it and returns its path.</summary>
    public static async Task<string> DownloadModelAsync(VoiceIsolatorModel model, string? downloadToPath = null,
                                                        Action<DownloadProgress>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        string path = downloadToPath ?? VoiceIsolatorModels.GetDefaultCachePath(model);
        await ModelDownloader.DownloadFileAsync(VoiceIsolatorModels.GetDownloadUrl(model), path, reportProgress,
                                                VoiceIsolatorModels.GetFileSize(model), cancellationToken).ConfigureAwait(false);
        return path;
    }

    /// <summary>
    /// Isolates the voice of mono audio at <paramref name="sampleRate"/>; returns it at the input rate unless
    /// <paramref name="outputSampleRate"/> is given (the length then matches the input exactly). Offline, long audio runs
    /// in overlapping chunks; in streaming mode the stream itself runs over the clip, half a second at a time, exactly as a
    /// live session would. Subtract the result from the input to get the removed background.
    /// </summary>
    public float[] Isolate(ReadOnlySpan<float> audio, int sampleRate, int? outputSampleRate = null, double? chunkSeconds = null,
                           double? overlapSeconds = null)
    {
        if (audio.IsEmpty) throw new ArgumentException("The audio is empty: there is nothing to isolate.", nameof(audio));
        foreach (float x in audio)
            if (!float.IsFinite(x)) throw new ArgumentException("The audio contains NaN or infinite samples.", nameof(audio));
        float[] input = Resampler.Resample(audio.ToArray(), sampleRate, SampleRate);
        float[] voice;
        if (Mode == IsolationMode.Streaming)
        {
            voice = RunStream(input);
        }
        else
        {
            double? chunk = chunkSeconds ?? Options.ChunkSeconds, overlap = overlapSeconds ?? Options.OverlapSeconds;
            voice = Network.Enhance(input, chunk, overlap);
            for (int pass = 1; pass < OfflinePasses; pass++) voice = Network.Enhance(voice, chunk, overlap);
        }
        voice = FitLength(voice, input.Length);
        int outRate = outputSampleRate is > 0 ? outputSampleRate.Value : sampleRate;
        voice = Resampler.Resample(voice, SampleRate, outRate);
        return outRate == sampleRate ? FitLength(voice, audio.Length) : voice;
    }

    /// <summary>Reads a WAV file and isolates its voice (mixed to mono); returns the voice and its sample rate.</summary>
    public (float[] Voice, float[] Input, int SampleRate) IsolateFile(string path, int? outputSampleRate = null)
    {
        var (samples, rate) = WaveFile.Read(path);
        var voice = Isolate(samples, rate, outputSampleRate);
        return (voice, samples, outputSampleRate is > 0 ? outputSampleRate.Value : rate);
    }

    /// <summary>Starts a streaming session (streaming checkpoints only): push 48 kHz mono samples in pieces of any size,
    /// receive the samples that became final, and <see cref="VoiceIsolatorStream.Flush"/> at the end.</summary>
    public VoiceIsolatorStream CreateStream(int? blockFrames = null, int? cacheFrames = null, int? windowFrames = null,
                                            int? crossfadeFrames = null)
    {
        if (Mode != IsolationMode.Streaming)
            throw new InvalidOperationException("Streaming needs the realtime checkpoint (voice-isolator-v1-realtime).");
        int window = windowFrames ?? Options.WindowFrames;
        int? crossfade = window == 0 ? null : crossfadeFrames ?? Options.CrossfadeFrames;
        return new VoiceIsolatorStream(Network, blockFrames ?? Options.BlockFrames, cacheFrames ?? Options.CacheFrames,
                                       window == 0 ? null : window, crossfade);
    }

    private float[] RunStream(float[] input)
    {
        using var stream = CreateStream();
        int piece = (int)(StreamPieceSeconds * SampleRate);
        var output = new List<float>(input.Length + SampleRate);
        for (int i = 0; i < input.Length; i += piece) output.AddRange(stream.Push(input.AsSpan(i, Math.Min(piece, input.Length - i))));
        output.AddRange(stream.Flush());
        return output.ToArray();
    }

    private static float[] FitLength(float[] x, int n)
    {
        if (x.Length == n) return x;
        var y = new float[n];
        x.AsSpan(0, Math.Min(n, x.Length)).CopyTo(y);
        return y;
    }

    public void Dispose() => Network.Dispose();
}
