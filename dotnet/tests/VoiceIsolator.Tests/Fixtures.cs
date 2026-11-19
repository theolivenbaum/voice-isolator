using System.Text.Json;
using VoiceIsolation.Model;
using Xunit.Abstractions;

namespace VoiceIsolation.Tests;

/// <summary>
/// Locates the checkpoints, the layer dumps of <c>dotnet/scripts/dump_reference.py</c> and the repository's golden
/// fixtures:
/// <list type="bullet">
/// <item><c>VOICE_ISOLATOR_MODELS_DIR</c>: a directory holding <c>denoiser.safetensors</c> and
/// <c>denoiser_realtime.safetensors</c> (default: the repository's <c>models/</c>)</item>
/// <item><c>VOICE_ISOLATOR_REFERENCE_DIR</c>: the reference dumps, one sub-directory per run (default:
/// <c>staging/dotnet-ref</c>)</item>
/// </list>
/// Tests that need them return early, and say so, when they are missing, so the kernel tests still run anywhere.
/// </summary>
public static class Fixtures
{
    /// <summary>The repository root (the directory holding <c>tests/fixtures</c>), found by walking up from the test
    /// assembly.</summary>
    public static string? RepositoryRoot { get; } = FindRoot();

    private static string? FindRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "tests", "fixtures", "offline_3s", "meta.json"))) return dir.FullName;
        return null;
    }

    public static string? ModelsDir => Existing(Environment.GetEnvironmentVariable("VOICE_ISOLATOR_MODELS_DIR"))
                                       ?? Existing(RepositoryRoot == null ? null : Path.Combine(RepositoryRoot, "models"));

    public static string? ReferenceDir => Existing(Environment.GetEnvironmentVariable("VOICE_ISOLATOR_REFERENCE_DIR"))
                                          ?? Existing(RepositoryRoot == null ? null : Path.Combine(RepositoryRoot, "staging", "dotnet-ref"));

    public static string? GoldenDir => Existing(RepositoryRoot == null ? null : Path.Combine(RepositoryRoot, "tests", "fixtures"));

    private static string? Existing(string? p) => !string.IsNullOrEmpty(p) && (File.Exists(p) || Directory.Exists(p)) ? p : null;

    public static string? ModelPath(VoiceIsolatorModel model)
    {
        string? dir = ModelsDir;
        if (dir == null) return null;
        string path = Path.Combine(dir, VoiceIsolatorModels.GetFileName(model));
        return File.Exists(path) ? path : null;
    }

    public static bool HaveModel(ITestOutputHelper output, VoiceIsolatorModel model)
    {
        if (ModelPath(model) != null) return true;
        output.WriteLine($"SKIPPED: {VoiceIsolatorModels.GetFileName(model)} not found; set VOICE_ISOLATOR_MODELS_DIR (see dotnet/README.md, 'Validation').");
        return false;
    }

    public static string? Reference(ITestOutputHelper output, string run)
    {
        string? dir = ReferenceDir == null ? null : Path.Combine(ReferenceDir, run);
        if (dir != null && File.Exists(Path.Combine(dir, "meta.json"))) return dir;
        output.WriteLine($"SKIPPED: no reference dump '{run}'; run dotnet/scripts/dump_reference.py and set VOICE_ISOLATOR_REFERENCE_DIR.");
        return null;
    }

    public static Npy Load(string dir, string name) => Npy.Load(Path.Combine(dir, name + ".npy"));

    public static JsonElement Meta(string dir) => JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json"))).RootElement;

    /// <summary>Asserts closeness and reports max-abs and relative-L2 error.</summary>
    public static (double RelL2, double MaxAbs) AssertClose(ITestOutputHelper output, string label, ReadOnlySpan<float> expected,
                                                            ReadOnlySpan<float> actual, double maxRelL2, double maxAbs = double.PositiveInfinity)
    {
        Assert.Equal(expected.Length, actual.Length);
        var (rel, mx) = Error(expected, actual);
        output.WriteLine($"{label,-24} n={expected.Length,10} maxAbs={mx:E2} relL2={rel:E2}");
        Assert.True(rel <= maxRelL2, $"{label}: relative L2 error {rel:E3} > {maxRelL2:E1}");
        Assert.True(mx <= maxAbs, $"{label}: max abs error {mx:E3} > {maxAbs:E1}");
        return (rel, mx);
    }

    public static (double RelL2, double MaxAbs) Error(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        double num = 0, den = 0, mx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
            mx = Math.Max(mx, Math.Abs(d));
        }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), mx);
    }

    /// <summary><c>[a, b, inner]</c> to <c>[b, a, inner]</c>.</summary>
    public static float[] Swap(ReadOnlySpan<float> x, int a, int b, int inner)
    {
        var y = new float[x.Length];
        for (int i = 0; i < a; i++)
            for (int j = 0; j < b; j++)
                x.Slice((i * b + j) * inner, inner).CopyTo(y.AsSpan((j * a + i) * inner, inner));
        return y;
    }
}

/// <summary>Loads each released network once for all tests in the "model" collection.</summary>
public sealed class ModelFixture : IDisposable
{
    private readonly Dictionary<VoiceIsolatorModel, Lazy<VoiceIsolatorNetwork?>> _networks = Enum.GetValues<VoiceIsolatorModel>()
        .ToDictionary(m => m, m => new Lazy<VoiceIsolatorNetwork?>(() => Fixtures.ModelPath(m) is { } p ? VoiceIsolatorNetwork.Load(p) : null));

    public VoiceIsolatorNetwork? Network(VoiceIsolatorModel model) => _networks[model].Value;

    public void Dispose()
    {
        foreach (var n in _networks.Values)
            if (n.IsValueCreated) n.Value?.Dispose();
    }
}

[CollectionDefinition("model")]
public sealed class ModelCollection : ICollectionFixture<ModelFixture>;
