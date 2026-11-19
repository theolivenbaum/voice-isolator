using System.Reflection;

namespace VoiceIsolation;

/// <summary>The released Voice Isolator v1 checkpoints.</summary>
public enum VoiceIsolatorModel
{
    /// <summary><c>voice-isolator-v1</c>: recorded audio, best quality; offline, every frame sees its whole 8 s chunk
    /// (64M parameters, 256 MB).</summary>
    Offline,

    /// <summary><c>voice-isolator-v1-realtime</c>: live audio; a causal stream with 0, 20 or 40 ms of look-ahead
    /// (58M parameters, 233 MB).</summary>
    Realtime,
}

/// <summary>
/// Names, Hub locations, checksums and the model packages of the <see cref="VoiceIsolatorModel"/> checkpoints (the Python
/// package's <c>registry.MODELS</c>). A model is found, in order, in its model package
/// (<c>VoiceIsolatorSharp.Models.Offline</c> / <c>.Realtime</c>, an embedded <c>.vimodel</c> resource), in the cache
/// directory, or downloaded from the pinned Hugging Face revision; every source is checked against the release checksum.
/// </summary>
public static class VoiceIsolatorModels
{
    /// <summary>The Hugging Face repository of the released weights.</summary>
    public const string Repository = "inworld/voice-isolator";

    /// <summary>The pinned Hub revision of the released weights.</summary>
    public const string Revision = "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970";

    /// <summary>The release name: <c>voice-isolator-v1</c> or <c>voice-isolator-v1-realtime</c>.</summary>
    public static string GetName(VoiceIsolatorModel model) => model switch
    {
        VoiceIsolatorModel.Offline => "voice-isolator-v1",
        VoiceIsolatorModel.Realtime => "voice-isolator-v1-realtime",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>File name of the checkpoint in the Hub repository.</summary>
    public static string GetFileName(VoiceIsolatorModel model) => model switch
    {
        VoiceIsolatorModel.Offline => "denoiser.safetensors",
        VoiceIsolatorModel.Realtime => "denoiser_realtime.safetensors",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>SHA-256 of the released checkpoint (<c>registry.py</c>).</summary>
    public static string GetSha256(VoiceIsolatorModel model) => model switch
    {
        VoiceIsolatorModel.Offline => "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53",
        VoiceIsolatorModel.Realtime => "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>Exact size in bytes of the released checkpoint.</summary>
    public static long GetFileSize(VoiceIsolatorModel model) => model switch
    {
        VoiceIsolatorModel.Offline => 256_260_128,
        VoiceIsolatorModel.Realtime => 232_780_800,
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>The checkpoint at its pinned revision on the Hub.</summary>
    public static string GetDownloadUrl(VoiceIsolatorModel model) =>
        $"https://huggingface.co/{Repository}/resolve/{Revision}/{GetFileName(model)}";

    /// <summary>The NuGet package (and assembly) that embeds the model: <c>VoiceIsolatorSharp.Models.Offline</c> or
    /// <c>VoiceIsolatorSharp.Models.Realtime</c>.</summary>
    public static string GetPackageId(VoiceIsolatorModel model) => model switch
    {
        VoiceIsolatorModel.Offline => "VoiceIsolatorSharp.Models.Offline",
        VoiceIsolatorModel.Realtime => "VoiceIsolatorSharp.Models.Realtime",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>Name of the embedded <c>.vimodel</c> resource in the model package.</summary>
    public static string GetResourceName(VoiceIsolatorModel model) => GetName(model) + IO.ModelArchive.Extension;

    /// <summary>Directory downloaded checkpoints are cached in: <c>$VOICE_ISOLATOR_CACHE</c>, else
    /// <c>~/.cache/voice-isolator</c>.</summary>
    public static string CacheDirectory =>
        Environment.GetEnvironmentVariable("VOICE_ISOLATOR_CACHE") is { Length: > 0 } c
            ? c
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "voice-isolator");

    /// <summary>Default local path of a downloaded checkpoint.</summary>
    public static string GetDefaultCachePath(VoiceIsolatorModel model) => Path.Combine(CacheDirectory, GetFileName(model));

    /// <summary>Opens the embedded <c>.vimodel</c> of the model package when it is referenced (or its assembly sits next to
    /// the application); null otherwise.</summary>
    public static Stream? TryOpenEmbedded(VoiceIsolatorModel model)
    {
        Assembly? assembly = AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(a => a.GetName().Name == GetPackageId(model));
        if (assembly == null)
        {
            try
            {
                assembly = Assembly.Load(new AssemblyName(GetPackageId(model)));
            }
            catch (Exception e) when (e is FileNotFoundException or FileLoadException or BadImageFormatException)
            {
                return null;
            }
        }
        return assembly.GetManifestResourceStream(GetResourceName(model));
    }

    /// <summary>Parses a model name: <c>offline</c> / <c>realtime</c> (or <c>streaming</c>), the release name, the
    /// checkpoint file name or the enum name (case-insensitive).</summary>
    public static bool TryParse(string? name, out VoiceIsolatorModel model)
    {
        model = default;
        if (string.IsNullOrWhiteSpace(name)) return false;
        string n = name.Trim();
        foreach (var m in Enum.GetValues<VoiceIsolatorModel>())
        {
            string file = GetFileName(m);
            string[] aliases = [m.ToString(), GetName(m), file, Path.GetFileNameWithoutExtension(file), GetPackageId(m),
                                m == VoiceIsolatorModel.Offline ? "offline" : "streaming"];
            if (aliases.Any(a => string.Equals(a, n, StringComparison.OrdinalIgnoreCase)))
            {
                model = m;
                return true;
            }
        }
        return false;
    }
}
