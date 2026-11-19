using System.Diagnostics;
using VoiceIsolation;
using VoiceIsolation.Audio;
using VoiceIsolation.IO;
using VoiceIsolation.Numerics;

return await Cli.RunAsync(args);

/// <summary>
/// <c>voice-isolate</c>: the Python package's command line in C#, plus <c>download</c>, <c>pack</c>, <c>info</c> and
/// <c>bench</c>.
/// </summary>
internal static class Cli
{
    private const string Usage = """
        voice-isolate <input.wav|dir> <output.wav|dir> [options]    isolate the voice
        voice-isolate download [--model <name>] [--out <path>]      fetch a released checkpoint (prints its path)
        voice-isolate pack --model <name|path> --out <file.vimodel> pack a checkpoint for a model package
        voice-isolate info [--model <name|path>]                    print a checkpoint's architecture
        voice-isolate bench [--model <name|path>] [--seconds 8] [--runs 3]

        Options (isolate):
          --model <name|path>     voice-isolator-v1 (offline), voice-isolator-v1-realtime, a .safetensors or .vimodel
                                  (default: the checkpoint of --mode)
          --mode offline|streaming  offline: every frame sees its whole chunk (best quality, the default);
                                  streaming: the causal stream, as a live session runs it (the realtime checkpoint)
          --lookahead <frames>    streaming look-ahead: 0, 2 or 4 frames of 10 ms (default: the checkpoint's, 2)
          --block-frames <n>      streaming: frames computed together (throughput only; default 4)
          --window-frames <n>     streaming: attention window (default 600; 0 turns windows off)
          --crossfade-frames <n>  streaming: lane crossfade before each window boundary (default 200)
          --chunk-seconds <s>     offline: seconds per render chunk (default: the checkpoint's, 8)
          --overlap-seconds <s>   offline: seconds two chunks overlap and crossfade (default 2)
          --output-sr <hz>        output sample rate (default 48000; 0 keeps each input's rate)
          --background <path>     offline, single file: where the removed background goes
                                  (default: <output-stem>_background.wav)
          --no-background         offline: write only the voice
          --float32               streaming: write float WAVs (default 16-bit PCM; offline always writes float)
          --threads <n>           worker threads (default: all cores)
        """;

    public static async Task<int> RunAsync(string[] argv)
    {
        try
        {
            if (argv.Length == 0 || argv[0] is "-h" or "--help")
            {
                Console.WriteLine(Usage);
                return argv.Length == 0 ? 2 : 0;
            }
            var (positional, options) = Parse(argv);
            return positional.FirstOrDefault() switch
            {
                "download" => await DownloadAsync(options),
                "pack" => await PackAsync(options),
                "info" => await InfoAsync(options),
                "bench" => await BenchAsync(options),
                _ => await IsolateAsync(positional, options),
            };
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or IOException or InvalidOperationException
                                       or NotSupportedException or HttpRequestException)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 1;
        }
    }

    private static readonly HashSet<string> Flags = ["--no-background", "--float32"];

    private static (List<string> Positional, Dictionary<string, string> Options) Parse(string[] argv)
    {
        var positional = new List<string>();
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < argv.Length; i++)
        {
            string a = argv[i];
            if (!a.StartsWith("--")) positional.Add(a);
            else if (Flags.Contains(a)) options[a] = "true";
            else if (i + 1 < argv.Length) options[a] = argv[++i];
            else throw new ArgumentException($"{a} needs a value.");
        }
        return (positional, options);
    }

    private static int? Int(Dictionary<string, string> o, string key) => o.TryGetValue(key, out var v) ? int.Parse(v) : null;
    private static double? Double(Dictionary<string, string> o, string key) =>
        o.TryGetValue(key, out var v) ? double.Parse(v, System.Globalization.CultureInfo.InvariantCulture) : null;

    private static string ModelName(Dictionary<string, string> o) =>
        o.TryGetValue("--model", out var m) ? m : o.TryGetValue("--mode", out var mode) && mode == "streaming"
            ? VoiceIsolatorModels.GetName(VoiceIsolatorModel.Realtime)
            : VoiceIsolatorModels.GetName(VoiceIsolatorModel.Offline);

    private static void Progress(DownloadProgress p)
    {
        if (p.TotalBytes is long total) Console.Error.Write($"\r{p.FileName}: {p.DownloadedBytes / 1e6:F0}/{total / 1e6:F0} MB ({p.Fraction:P0})   ");
    }

    private static Task<VoiceIsolator> LoadAsync(Dictionary<string, string> o, VoiceIsolatorOptions options)
    {
        string model = ModelName(o);
        var sw = Stopwatch.StartNew();
        var task = VoiceIsolator.LoadAsync(model, options, Progress);
        return task.ContinueWith(t =>
        {
            var isolator = t.GetAwaiter().GetResult();
            var cfg = isolator.Network.Config;
            Console.Error.WriteLine($"\rloaded {model} ({isolator.Mode.ToString().ToLowerInvariant()}, {isolator.Network.ParameterCount / 1e6:F1}M parameters, " +
                                    $"{cfg.BlockTypes.Length} blocks) in {sw.Elapsed.TotalSeconds:F1} s");
            return isolator;
        });
    }

    private static async Task<int> IsolateAsync(List<string> positional, Dictionary<string, string> o)
    {
        if (positional.Count != 2) throw new ArgumentException("Give an input and an output (see --help).");
        string mode = o.GetValueOrDefault("--mode", "offline");
        if (mode is not ("offline" or "streaming")) throw new ArgumentException("--mode must be offline or streaming.");
        int outputRate = Int(o, "--output-sr") ?? 48000;
        if (outputRate < 0) throw new ArgumentException("--output-sr must be positive, or 0 to keep the input rate.");
        if (o.ContainsKey("--background") && mode != "offline") throw new ArgumentException("--background is available only in offline mode.");
        bool background = mode == "offline" && !o.ContainsKey("--no-background");

        var jobs = Jobs(positional[0], positional[1], background, o.GetValueOrDefault("--background"));
        Console.WriteLine($"{jobs.Count} file(s)");
        var options = new VoiceIsolatorOptions
        {
            Threads = Int(o, "--threads") ?? 0,
            LookaheadFrames = Int(o, "--lookahead"),
            BlockFrames = Int(o, "--block-frames") ?? 4,
            WindowFrames = Int(o, "--window-frames") ?? 600,
            CrossfadeFrames = Int(o, "--crossfade-frames") ?? 200,
            ChunkSeconds = Double(o, "--chunk-seconds"),
            OverlapSeconds = Double(o, "--overlap-seconds"),
        };
        using var isolator = await LoadAsync(o, options);
        if ((isolator.Mode == IsolationMode.Streaming) != (mode == "streaming"))
            throw new ArgumentException($"The checkpoint runs {isolator.Mode.ToString().ToLowerInvariant()}; pass --mode {isolator.Mode.ToString().ToLowerInvariant()}.");
        var total = Stopwatch.StartNew();
        int ok = 0;
        foreach (var (src, dst, bg) in jobs)
        {
            try
            {
                var sw = Stopwatch.StartNew();
                var (x, rate) = WaveFile.Read(src);
                int outRate = outputRate == 0 ? rate : outputRate;
                float[] voice = isolator.Isolate(x, rate, outRate);
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dst))!);
                if (bg != null)
                {
                    float[] aligned = Resampler.Resample(x, rate, outRate);
                    var removed = new float[voice.Length];
                    for (int i = 0; i < removed.Length; i++) removed[i] = (i < aligned.Length ? aligned[i] : 0f) - voice[i];
                    WaveFile.WriteMonoFloat(dst, voice, outRate);
                    WaveFile.WriteMonoFloat(bg, removed, outRate);
                }
                else if (o.ContainsKey("--float32") || mode == "offline") WaveFile.WriteMonoFloat(dst, voice, outRate);
                else WaveFile.WriteMono16(dst, voice, outRate);
                double seconds = (double)x.Length / rate;
                Console.WriteLine($"{src} -> {dst}{(bg != null ? $" (+ {bg})" : "")}  {seconds:F1} s of audio in {sw.Elapsed.TotalSeconds:F1} s " +
                                  $"(RTF {sw.Elapsed.TotalSeconds / seconds:F2})");
                ok++;
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                Console.WriteLine($"FAIL {src} {e.Message}");
            }
        }
        Console.WriteLine($"DONE {ok}/{jobs.Count}  ({total.Elapsed.TotalSeconds:F0}s)");
        return ok == jobs.Count ? 0 : 1;
    }

    /// <summary>(input, voice, background) per file: a file maps to the output path, a directory's WAVs to
    /// <c>&lt;output&gt;/&lt;stem&gt;.wav</c>.</summary>
    private static List<(string Src, string Dst, string? Background)> Jobs(string input, string output, bool background, string? backgroundPath)
    {
        string? Bg(string dst) => background ? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(dst))!, Path.GetFileNameWithoutExtension(dst) + "_background.wav") : null;
        if (Directory.Exists(input))
        {
            if (backgroundPath != null) throw new ArgumentException("--background takes a single-file input.");
            return Directory.EnumerateFiles(input, "*.wav", SearchOption.AllDirectories).Order(StringComparer.Ordinal)
                .Select(f => Path.Combine(output, Path.ChangeExtension(Path.GetRelativePath(input, f), ".wav")))
                .Zip(Directory.EnumerateFiles(input, "*.wav", SearchOption.AllDirectories).Order(StringComparer.Ordinal), (dst, src) => (src, dst, Bg(dst)))
                .ToList();
        }
        if (!File.Exists(input)) throw new FileNotFoundException($"No such file or directory: {input}");
        return [(input, output, background ? backgroundPath ?? Bg(output) : null)];
    }

    private static async Task<int> DownloadAsync(Dictionary<string, string> o)
    {
        string name = o.GetValueOrDefault("--model", VoiceIsolatorModels.GetName(VoiceIsolatorModel.Offline));
        if (!VoiceIsolatorModels.TryParse(name, out var model)) throw new ArgumentException($"Unknown model '{name}'.");
        string path = await VoiceIsolator.DownloadModelAsync(model, o.GetValueOrDefault("--out"), Progress);
        Console.Error.WriteLine();
        Console.WriteLine(path);
        return 0;
    }

    /// <summary>Packs a released (downloaded, checksum-verified) or local checkpoint into a <c>.vimodel</c>.</summary>
    private static async Task<int> PackAsync(Dictionary<string, string> o)
    {
        string name = o.GetValueOrDefault("--model") ?? throw new ArgumentException("pack needs --model.");
        string output = o.GetValueOrDefault("--out") ?? throw new ArgumentException("pack needs --out.");
        string source = name;
        if (!File.Exists(name))
        {
            if (!VoiceIsolatorModels.TryParse(name, out var model)) throw new ArgumentException($"Unknown model '{name}'.");
            source = await VoiceIsolator.DownloadModelAsync(model, null, Progress);
            Console.Error.WriteLine();
            using var stream = File.OpenRead(source);
            string sha = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(stream));
            if (sha != VoiceIsolatorModels.GetSha256(model)) throw new InvalidDataException($"{source}: checksum mismatch ({sha}).");
        }
        var sw = Stopwatch.StartNew();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        string temp = output + ".partial";
        using (var file = File.Create(temp)) ModelArchive.Pack(source, file);
        // Unpack once to prove the archive restores the checkpoint byte for byte.
        using (var check = File.OpenRead(temp)) ModelArchive.Unpack(check).Dispose();
        File.Move(temp, output, overwrite: true);
        Console.WriteLine($"{source} ({new FileInfo(source).Length / 1e6:F1} MB) -> {output} ({new FileInfo(output).Length / 1e6:F1} MB) in {sw.Elapsed.TotalSeconds:F0} s");
        return 0;
    }

    private static async Task<int> InfoAsync(Dictionary<string, string> o)
    {
        using var isolator = await LoadAsync(o, new VoiceIsolatorOptions { Threads = 1 });
        var c = isolator.Network.Config;
        Console.WriteLine($"mode                {isolator.Mode.ToString().ToLowerInvariant()}");
        Console.WriteLine($"parameters          {isolator.Network.ParameterCount:N0}");
        Console.WriteLine($"blocks              {string.Join(", ", c.BlockTypes)}");
        Console.WriteLine($"width / heads       {c.Width} / {c.NumHeads}");
        Console.WriteLine($"bands               {isolator.Network.Bands}");
        Console.WriteLine($"stft                n_fft {c.NFft}, hop {c.HopLength} at {c.SampleRate} Hz");
        Console.WriteLine($"chunk / overlap     {c.ChunkSeconds} s / {c.OverlapSeconds} s");
        if (isolator.Mode == IsolationMode.Streaming)
        {
            Console.WriteLine($"look-ahead          {isolator.LookaheadFrames} frames (trained: {string.Join(", ", isolator.LookaheadChoices)})");
            Console.WriteLine($"attention window    {c.AttentionWindowFrames} frames");
            Console.WriteLine($"latency             {isolator.LatencySamples} samples ({1000.0 * isolator.LatencySamples / c.SampleRate:F0} ms)");
        }
        foreach (var (k, v) in isolator.Network.Metadata.Where(kv => kv.Key != "config")) Console.WriteLine($"{k,-20}{v}");
        return 0;
    }

    private static async Task<int> BenchAsync(Dictionary<string, string> o)
    {
        double seconds = Double(o, "--seconds") ?? 8;
        int runs = Int(o, "--runs") ?? 3;
        using var isolator = await LoadAsync(o, new VoiceIsolatorOptions { Threads = Int(o, "--threads") ?? 0 });
        var rng = new Random(0);
        var x = new float[(int)(seconds * isolator.SampleRate)];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        isolator.Isolate(x, isolator.SampleRate);
        var times = new List<double>();
        Profiler.Reset();
        for (int r = 0; r < runs; r++)
        {
            var sw = Stopwatch.StartNew();
            isolator.Isolate(x, isolator.SampleRate);
            times.Add(sw.Elapsed.TotalSeconds);
            Console.WriteLine($"run {r + 1}: {times[^1]:F2} s (RTF {times[^1] / seconds:F3})");
        }
        times.Sort();
        Console.WriteLine($"median: {times[times.Count / 2]:F2} s for {seconds} s of audio (RTF {times[times.Count / 2] / seconds:F3}), " +
                          $"{Environment.ProcessorCount} cores, {(Vector512.IsHardwareAccelerated ? "AVX-512" : Vector256.IsHardwareAccelerated ? "AVX2" : "scalar")}");
        if (Profiler.Enabled) Console.WriteLine(Profiler.Report());
        return 0;
    }
}
