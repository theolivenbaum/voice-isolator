using Xunit.Abstractions;

namespace VoiceIsolation.Tests;

/// <summary>
/// The repository's golden fixtures (<c>tests/fixtures</c>): renders of the independent training reference, never of the
/// Python package, bound sample by sample at the tolerance in each <c>meta.json</c> (1e-4), exactly as
/// <c>tests/inworld/voice_isolation/test_golden.py</c> checks the Python package.
/// </summary>
[Collection("model")]
public sealed class GoldenTests(ModelFixture fixture, ITestOutputHelper output)
{
    [Theory]
    [InlineData("offline_3s")]
    [InlineData("offline_11s")]
    [InlineData("realtime_stream_3s")]
    [InlineData("realtime_stream_11s")]
    public void MatchesGolden(string fixtureName)
    {
        if (Fixtures.GoldenDir is not { } root)
        {
            output.WriteLine("SKIPPED: tests/fixtures not found above the test assembly.");
            return;
        }
        string dir = Path.Combine(root, fixtureName);
        var meta = Fixtures.Meta(dir);
        var model = meta.GetProperty("mode").GetString() == "streaming" ? VoiceIsolatorModel.Realtime : VoiceIsolatorModel.Offline;
        if (!Fixtures.HaveModel(output, model)) return;
        string sha = meta.GetProperty("model_sha256").GetString()!;
        Assert.Equal(VoiceIsolatorModels.GetSha256(model), sha);
        var net = fixture.Network(model)!;
        double tolerance = meta.GetProperty("tolerance").GetDouble();
        var (input, rate) = Audio.WaveFile.Read(Path.Combine(dir, "input.wav"));
        float[] voice;
        if (meta.TryGetProperty("runner", out var runner))
        {
            Assert.Equal(48000, rate);
            net.SetLookahead(meta.GetProperty("lookahead_frames").GetInt32());
            using var stream = new Model.VoiceIsolatorStream(net, runner.GetProperty("block_frames").GetInt32(),
                                                              runner.GetProperty("cache_frames").GetInt32(),
                                                              runner.GetProperty("window_frames").GetInt32(),
                                                              runner.GetProperty("crossfade_frames").GetInt32());
            int[] pushes = runner.GetProperty("push").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var result = new List<float>();
            int at = 0, k = 0;
            while (at < input.Length)
            {
                int n = Math.Min(pushes[k++ % pushes.Length], input.Length - at);
                result.AddRange(stream.Push(input.AsSpan(at, n)));
                at += n;
            }
            result.AddRange(stream.Flush());
            voice = result.ToArray();
        }
        else
        {
            var isolator = VoiceIsolator.Create(net);
            voice = isolator.Isolate(input, rate, 48000, meta.GetProperty("chunk_seconds").GetDouble(), meta.GetProperty("overlap_seconds").GetDouble());
        }
        float[] expected = Npy.Load(Path.Combine(dir, "expected.npy")).Data;
        Assert.Equal(expected.Length, voice.Length);
        Fixtures.AssertClose(output, fixtureName, expected, voice, 1e-5, tolerance);
    }
}
