using VoiceIsolation.Model;
using Xunit.Abstractions;

namespace VoiceIsolation.Tests;

/// <summary>The public API over the released checkpoints: rates and lengths, concurrency, and streaming invariants.</summary>
[Collection("model")]
public sealed class ApiTests(ModelFixture fixture, ITestOutputHelper output)
{
    private static float[] Clip(int samples, int seed = 7)
    {
        // a few harmonics under noise, so the network has a voice-like signal to keep
        var rng = new Random(seed);
        var x = new float[samples];
        for (int i = 0; i < samples; i++)
            x[i] = 0.1f * MathF.Sin(2 * MathF.PI * 220 * i / 48000f) + 0.05f * MathF.Sin(2 * MathF.PI * 440 * i / 48000f) +
                   0.02f * (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    [Fact]
    public void IsolateKeepsTheInputRateAndLength()
    {
        if (!Fixtures.HaveModel(output, VoiceIsolatorModel.Offline)) return;
        var isolator = VoiceIsolator.Create(fixture.Network(VoiceIsolatorModel.Offline)!);
        float[] x16 = Audio.Resampler.Resample(Clip(48000), 48000, 16000);
        float[] y16 = isolator.Isolate(x16, 16000);
        Assert.Equal(x16.Length, y16.Length);
        float[] y48 = isolator.Isolate(x16, 16000, outputSampleRate: 48000);
        Assert.Equal(48000, y48.Length);
        Assert.All(y16, v => Assert.True(float.IsFinite(v)));
        Assert.Throws<ArgumentException>(() => isolator.Isolate([], 48000));
        Assert.Throws<ArgumentException>(() => isolator.Isolate([0f, float.NaN], 48000));
    }

    [Fact]
    public void ConcurrentCallsMatchSequentialOnes()
    {
        if (!Fixtures.HaveModel(output, VoiceIsolatorModel.Offline)) return;
        var net = fixture.Network(VoiceIsolatorModel.Offline)!;
        float[][] clips = [Clip(24000, 1), Clip(30000, 2), Clip(20000, 3)];
        float[][] sequential = clips.Select(c => net.Forward(c)).ToArray();
        var concurrent = new float[clips.Length][];
        Parallel.For(0, clips.Length, i => concurrent[i] = net.Forward(clips[i]));
        for (int i = 0; i < clips.Length; i++) Assert.Equal(sequential[i], concurrent[i]);
    }

    [Fact]
    public void WindowedStreamDoesNotDependOnThePushes()
    {
        // With windows and a cache that never trims, a frame's context is fixed by the windows: the output is the same
        // for any push sizes (streaming.Stream's documented property), here across a window boundary.
        if (!Fixtures.HaveModel(output, VoiceIsolatorModel.Realtime)) return;
        var net = fixture.Network(VoiceIsolatorModel.Realtime)!;
        net.SetLookahead(2);
        // the fixture clip (speech in hiss) when present: the network keeps speech, so the output is not near silence
        string wav = Path.Combine(Fixtures.GoldenDir ?? "", "offline_3s", "input.wav");
        float[] x = File.Exists(wav) ? Audio.WaveFile.Read(wav).Samples[..(48000 * 2)] : Clip(48000 * 2);
        float[] Run(int[] pushes)
        {
            using var stream = new VoiceIsolatorStream(net, blockFrames: 4, cacheFrames: null, windowFrames: 120, crossfadeFrames: 40);
            var y = new List<float>();
            int at = 0, k = 0;
            while (at < x.Length)
            {
                int n = Math.Min(pushes[k++ % pushes.Length], x.Length - at);
                y.AddRange(stream.Push(x.AsSpan(at, n)));
                at += n;
            }
            y.AddRange(stream.Flush());
            Assert.Equal(new[] { 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200 }.Where(b => b <= x.Length / 480), stream.Boundaries);
            return y.ToArray();
        }
        float[] a = Run([960]), b = Run([7001, 13, 480, 33333]);
        Assert.Equal(x.Length, a.Length);
        Fixtures.AssertClose(output, "pushes 960 vs mixed", a, b, 1e-5, 1e-6);
    }

    [Fact]
    public void StreamingIsolateIsTheHalfSecondStream()
    {
        if (!Fixtures.HaveModel(output, VoiceIsolatorModel.Realtime)) return;
        var net = fixture.Network(VoiceIsolatorModel.Realtime)!;
        var isolator = VoiceIsolator.Create(net, new VoiceIsolatorOptions { LookaheadFrames = 4 });
        Assert.Equal(IsolationMode.Streaming, isolator.Mode);
        Assert.Equal(2048 + 4 * 480, isolator.LatencySamples);
        float[] x = Clip(30000);
        float[] viaIsolate = isolator.Isolate(x, 48000);
        using var stream = isolator.CreateStream();
        var manual = new List<float>(stream.Push(x.AsSpan(0, 24000)));
        manual.AddRange(stream.Push(x.AsSpan(24000)));
        manual.AddRange(stream.Flush());
        Assert.Equal(manual.ToArray(), viaIsolate);
        isolator.LookaheadFrames = 2;
        Assert.Throws<ArgumentException>(() => isolator.LookaheadFrames = 3);
    }
}
