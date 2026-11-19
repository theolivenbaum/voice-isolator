using VoiceIsolation.IO;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>
/// The per-band encoders and decoder heads. Each band's slice of the spectrum, <c>(re, im)</c> per bin, becomes one token
/// per frame (<c>LayerNorm → Linear</c>, plus the band's embedding and the look-ahead embedding); each band's head turns
/// its tokens back into four numbers per bin (<c>FiLM → LayerNorm → Linear → GELU → Linear</c>): the complex mask delta
/// and the residual.
/// </summary>
internal sealed unsafe class BandCodec : IDisposable
{
    public BandCodec(SafeTensorsFile f, VoiceIsolatorConfig config, (int Start, int End)[] edges)
    {
        Width = config.Width;
        Edges = edges;
        Bins = config.Bins;
        int bands = edges.Length;
        EncoderNorms = new Norm[bands];
        Encoders = new Linear[bands];
        HeadNorms = new Norm[bands];
        Heads1 = new Linear[bands];
        Heads2 = new Linear[bands];
        for (int b = 0; b < bands; b++)
        {
            int w = edges[b].End - edges[b].Start;
            EncoderNorms[b] = Norm.Load(f, $"encoders.{b}.0", 2 * w);
            Encoders[b] = Linear.Load(f, $"encoders.{b}.1", Width, 2 * w);
            HeadNorms[b] = Norm.Load(f, $"denoise_decoder.heads.{b}.0", Width);
            Heads1[b] = Linear.Load(f, $"denoise_decoder.heads.{b}.1", Width, Width);
            Heads2[b] = Linear.Load(f, $"denoise_decoder.heads.{b}.3", 4 * w, Width);
        }
        BandEmbedding = new Vec(new ReadOnlySpan<float>(f.Float("band_embedding", bands, Width), bands * Width));
        if (config.LookaheadFramesChoices.Length > 0)
        {
            int positions = config.LookaheadFramesChoices.Length;
            LookaheadEmbedding = new Vec(new ReadOnlySpan<float>(f.Float("lookahead_embedding.weight", positions, Width), positions * Width));
        }

        // FiLM of the decoder: Linear(2 → 2·width) of the fixed controls (1, 0); scale and shift are constants.
        float* cw = f.Float("denoise_decoder.condition.weight", 2 * Width, 2);
        float* cb = f.Float("denoise_decoder.condition.bias", 2 * Width);
        FilmScale = new Vec(Width);
        FilmShift = new Vec(Width);
        for (int j = 0; j < Width; j++)
        {
            float scale = cw[2 * j] * 1f + cw[2 * j + 1] * 0f + cb[j];
            float shift = cw[2 * (Width + j)] * 1f + cw[2 * (Width + j) + 1] * 0f + cb[Width + j];
            FilmScale.Ptr[j] = 1f + scale;
            FilmShift.Ptr[j] = shift;
        }
        EncoderBias = new Vec(bands * Width);
        SetLookahead(-1);
    }

    public int Width { get; }
    public int Bins { get; }
    public (int Start, int End)[] Edges { get; }
    public Norm[] EncoderNorms { get; }
    public Linear[] Encoders { get; }
    public Norm[] HeadNorms { get; }
    public Linear[] Heads1 { get; }
    public Linear[] Heads2 { get; }
    public Vec BandEmbedding { get; }
    public Vec? LookaheadEmbedding { get; }
    public Vec FilmScale { get; }
    public Vec FilmShift { get; }

    /// <summary>Each band's encoder bias with its band embedding and the look-ahead embedding folded in.</summary>
    public Vec EncoderBias { get; }

    /// <summary>Folds the look-ahead embedding of position <paramref name="index"/> (none when negative) into the
    /// encoder biases.</summary>
    public void SetLookahead(int index)
    {
        int W = Width;
        for (int b = 0; b < Edges.Length; b++)
        {
            float* dst = EncoderBias.Ptr + b * W, bias = Encoders[b].Bias, emb = BandEmbedding.Ptr + b * W;
            float* la = index >= 0 && LookaheadEmbedding != null ? LookaheadEmbedding.Ptr + index * W : null;
            for (int j = 0; j < W; j++) dst[j] = bias[j] + emb[j] + (la != null ? la[j] : 0f);
        }
    }

    /// <summary>
    /// Band tokens <c>[bands, frames, width]</c> of the spectra <paramref name="spectrum"/> <c>[frames, bins, 2]</c>
    /// (<c>model.encode</c>): rows <paramref name="ldSpectrum"/> floats apart.
    /// </summary>
    public void Encode(float* spectrum, long ldSpectrum, int frames, float* features, Workspace ws, Parallelism par)
    {
        int W = Width, bands = Edges.Length;
        int maxIn = Edges.Max(e => 2 * (e.End - e.Start));
        float* scratch = ws.Get(Slot.Norm, (long)bands * frames * maxIn);
        par.For(bands, b =>
        {
            int start = Edges[b].Start, n = 2 * (Edges[b].End - start);
            float* normed = scratch + (long)b * frames * maxIn;
            EncoderNorms[b].Forward(spectrum + 2 * start, ldSpectrum, normed, maxIn, frames);
            Encoders[b].Forward(normed, maxIn, frames, features + (long)b * frames * W, W, Parallelism.Single,
                                biasOverride: EncoderBias.Ptr + b * W);
        });
    }

    /// <summary>
    /// The decoder's raw output <c>[frames, bins, 4]</c> of the features <c>[bands, frames, width]</c>
    /// (<c>_Decoder.forward</c>): rows <paramref name="ldRaw"/> floats apart.
    /// </summary>
    public void Decode(float* features, int frames, float* raw, long ldRaw, Workspace ws, Parallelism par)
    {
        int W = Width, bands = Edges.Length;
        float* scratch = ws.Get(Slot.Hidden, (long)bands * frames * 2 * W);
        par.For(bands, b =>
        {
            float* x = features + (long)b * frames * W;
            float* h = scratch + (long)b * frames * 2 * W, h2 = h + (long)frames * W;
            for (int t = 0; t < frames; t++)
            {
                // features · (1 + scale) + shift, product then sum like the reference
                var src = new ReadOnlySpan<float>(x + (long)t * W, W);
                var dst = new Span<float>(h + (long)t * W, W);
                System.Numerics.Tensors.TensorPrimitives.Multiply(src, new ReadOnlySpan<float>(FilmScale.Ptr, W), dst);
                System.Numerics.Tensors.TensorPrimitives.Add(dst, new ReadOnlySpan<float>(FilmShift.Ptr, W), dst);
            }
            HeadNorms[b].Forward(h, h, frames, Parallelism.Single);
            Heads1[b].Forward(h, W, frames, h2, W, Parallelism.Single, Activation.Gelu);
            Heads2[b].Forward(h2, W, frames, raw + 4L * Edges[b].Start, ldRaw, Parallelism.Single);
        });
    }

    public void Dispose()
    {
        foreach (var d in EncoderNorms) d.Dispose();
        foreach (var d in Encoders) d.Dispose();
        foreach (var d in HeadNorms) d.Dispose();
        foreach (var d in Heads1) d.Dispose();
        foreach (var d in Heads2) d.Dispose();
        BandEmbedding.Dispose();
        LookaheadEmbedding?.Dispose();
        FilmScale.Dispose();
        FilmShift.Dispose();
        EncoderBias.Dispose();
    }
}
