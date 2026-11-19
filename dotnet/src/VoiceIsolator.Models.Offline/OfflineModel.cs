using System.Reflection;

namespace VoiceIsolation.Models;

/// <summary>The embedded <c>voice-isolator-v1</c> checkpoint (a packed <c>.vimodel</c>). <see cref="VoiceIsolator.LoadAsync(VoiceIsolatorModel, VoiceIsolatorOptions?, string?, Action{DownloadProgress}?, CancellationToken)"/>
/// finds it by itself when this package is referenced; <see cref="Load"/> loads it explicitly.</summary>
public static class OfflineModel
{
    /// <summary>Name of the manifest resource holding the packed checkpoint.</summary>
    public const string ResourceName = "voice-isolator-v1.vimodel";

    /// <summary>Whether this build of the package carries the weights.</summary>
    public static bool IsAvailable => typeof(OfflineModel).Assembly.GetManifestResourceInfo(ResourceName) != null;

    /// <summary>Opens the packed checkpoint.</summary>
    public static Stream Open() => typeof(OfflineModel).Assembly.GetManifestResourceStream(ResourceName)
        ?? throw new InvalidOperationException($"This build of {typeof(OfflineModel).Assembly.GetName().Name} carries no weights (packed without {ResourceName}).");

    /// <summary>Loads the embedded checkpoint, checked against the release SHA-256.</summary>
    public static VoiceIsolator Load(VoiceIsolatorOptions? options = null)
    {
        using var stream = Open();
        var network = Model.VoiceIsolatorNetwork.Load(stream, options?.Threads ?? 0, VoiceIsolatorModels.GetSha256(VoiceIsolatorModel.Offline), ResourceName);
        return VoiceIsolator.Create(network, options);
    }
}
