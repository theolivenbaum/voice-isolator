using VoiceIsolation.Model;
using Xunit.Abstractions;

namespace VoiceIsolation.Tests;

/// <summary>
/// Layer-by-layer parity with the PyTorch network (dumps of <c>dotnet/scripts/dump_reference.py</c>): the STFT, the band
/// encoders, the time half and output of every block (and each attention's projection and each forward Mamba layer), the
/// decoder heads, the correction and the output audio, for both released checkpoints over the same 3 s clip; and the
/// package's own stream over it for several push patterns.
/// </summary>
[Collection("model")]
public sealed class ParityTests(ModelFixture fixture, ITestOutputHelper output)
{
    // Relative L2 bounds per stage. Float32 GEMMs, softmax and the scan sum in a different order than PyTorch's kernels, so
    // the error grows slowly through the 7 blocks (to at most ~5e-6, AVX-512 or not); the golden fixtures bound the output
    // sample by sample as well.
    private const double StftBound = 1e-6;
    private const double LayerBound = 1e-5;
    private const double OutputBound = 2e-6;
    private const double OutputMaxAbs = 1e-5;

    [Theory]
    [InlineData(VoiceIsolatorModel.Offline, "offline")]
    [InlineData(VoiceIsolatorModel.Realtime, "realtime")]
    public void EveryLayerMatchesPyTorch(VoiceIsolatorModel model, string run)
    {
        if (!Fixtures.HaveModel(output, model) || Fixtures.Reference(output, run) is not { } dir) return;
        var net = fixture.Network(model)!;
        var meta = Fixtures.Meta(dir);
        net.SetLookaheadIfCausal(meta.GetProperty("lookahead_frames").GetInt32());
        float[] input = Fixtures.Load(dir, "input").Data;
        int frames = net.Spectral.Frames(input.Length), bands = net.Bands, W = net.Config.Width, bins = net.Config.Bins;

        var traced = new Dictionary<string, (float[] Data, TraceLayout Layout)>();
        float[] result = net.Forward(input, (stage, data, layout) => traced[stage] = (data.ToArray(), layout));

        var names = meta.GetProperty("arrays").EnumerateArray().Select(a => a.GetProperty("name").GetString()!).ToList();
        int compared = 0;
        foreach (string name in names)
        {
            if (name == "input" || name.StartsWith("stream_")) continue;
            Assert.True(traced.ContainsKey(name), $"stage {name} was not traced");
            var (data, layout) = traced[name];
            float[] expected = Fixtures.Load(dir, name).Data;
            float[] actual = name switch
            {
                // [frames, bins, k] here, [bins, frames, k] in the dump
                "stft" or "correction" => Fixtures.Swap(data, frames, bins, 2),
                "decoder" => Fixtures.Swap(data, frames, bins, 4),
                _ when layout == TraceLayout.FrameMajor => Fixtures.Swap(data, frames, bands, W),
                _ => data,
            };
            double bound = name == "stft" ? StftBound : name == "output" ? OutputBound : LayerBound;
            Fixtures.AssertClose(output, name, expected, actual, bound);
            compared++;
        }
        Fixtures.AssertClose(output, "forward result", Fixtures.Load(dir, "output").Data, result, OutputBound, OutputMaxAbs);
        Assert.True(compared >= 20, $"only {compared} stages compared");
    }

    [Theory]
    [InlineData("realtime", "stream_plain_960", 960, 0)]
    [InlineData("realtime", "stream_windowed_960", 960, 600)]
    [InlineData("realtime", "stream_windowed_7001", 7001, 600)]
    [InlineData("realtime_11s", "stream_plain_960", 960, 0)]
    [InlineData("realtime_11s", "stream_windowed_960", 960, 600)]
    [InlineData("realtime_11s", "stream_windowed_7001", 7001, 600)]
    public void StreamMatchesPyTorchStream(string run, string name, int piece, int windowFrames)
    {
        if (!Fixtures.HaveModel(output, VoiceIsolatorModel.Realtime) || Fixtures.Reference(output, run) is not { } dir) return;
        var net = fixture.Network(VoiceIsolatorModel.Realtime)!;
        net.SetLookahead(Fixtures.Meta(dir).GetProperty("lookahead_frames").GetInt32());
        float[] input = Fixtures.Load(dir, "input").Data;
        using var stream = new VoiceIsolatorStream(net, 4, null, windowFrames == 0 ? null : windowFrames, windowFrames == 0 ? null : 200);
        var result = new List<float>();
        for (int i = 0; i < input.Length; i += piece) result.AddRange(stream.Push(input.AsSpan(i, Math.Min(piece, input.Length - i))));
        result.AddRange(stream.Flush());
        Fixtures.AssertClose(output, $"{run}/{name}", Fixtures.Load(dir, name).Data, result.ToArray(), OutputBound, OutputMaxAbs);
    }
}

internal static class NetworkTestExtensions
{
    public static void SetLookaheadIfCausal(this VoiceIsolatorNetwork net, int frames)
    {
        if (!net.Offline) net.SetLookahead(frames);
    }
}
