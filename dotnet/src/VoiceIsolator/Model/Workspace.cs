using VoiceIsolation.Numerics;

namespace VoiceIsolation.Model;

/// <summary>Named scratch buffers for one forward pass, grown on demand and reused across layers and chunks.</summary>
internal sealed unsafe class Workspace : IDisposable
{
    private readonly Dictionary<Slot, NativeBuffer<float>> _buffers = new();

    /// <summary>Called with intermediate results when set (layer-by-layer validation); also turns off row grouping,
    /// so every traced tensor is whole.</summary>
    public TraceCallback? Trace { get; set; }

    /// <summary>Prefix of traced stage names (the block being run, e.g. <c>block3.</c>).</summary>
    public string TracePrefix { get; set; } = "";

    /// <summary>A buffer of at least <paramref name="length"/> floats; its contents are undefined.</summary>
    public float* Get(Slot slot, long length)
    {
        if (!_buffers.TryGetValue(slot, out var buffer) || buffer.Length < length)
        {
            buffer?.Dispose();
            buffer = new NativeBuffer<float>(Math.Max(length, 1024), zero: false);
            _buffers[slot] = buffer;
        }
        return buffer.Ptr;
    }

    public void Emit(string name, float* data, long length, TraceLayout layout = TraceLayout.BandMajor)
    {
        if (Trace != null) Trace(TracePrefix + name, new ReadOnlySpan<float>(data, checked((int)length)), layout);
    }

    public void Dispose()
    {
        foreach (var b in _buffers.Values) b.Dispose();
        _buffers.Clear();
    }
}

/// <summary>Scratch buffer names of <see cref="Workspace"/>.</summary>
internal enum Slot
{
    Features, Transposed, Spectrum, Raw, Corrected, Norm, Qkv, Context, Hidden, AttentionPack, Cat, Residual,
    MambaXz, MambaU, MambaXdbl, MambaDelta, MambaY, Ola, BlockSpectra, LaneB, RawB,
}

/// <summary>How a traced tensor's rows are ordered.</summary>
public enum TraceLayout
{
    /// <summary><c>[bands, frames, width]</c>: every band's frames in a row (the time paths).</summary>
    BandMajor,

    /// <summary><c>[frames, bands, width]</c>: every frame's bands in a row (the band paths).</summary>
    FrameMajor,

    /// <summary>Any other layout, documented by the stage.</summary>
    Other,
}

/// <summary>Receives a named intermediate result of the network (see <see cref="VoiceIsolatorNetwork.Forward(ReadOnlySpan{float}, TraceCallback?)"/>).</summary>
public delegate void TraceCallback(string stage, ReadOnlySpan<float> data, TraceLayout layout);

/// <summary>Rotary position tables <c>cos/sin[position, headDim / 2]</c> computed like <c>model.rope_tables</c> (float32
/// inverse frequencies and angles), shared by every attention block and grown on demand.</summary>
internal static unsafe class Rope
{
    private static readonly Lock Gate = new();
    private static float[] _cos = [], _sin = [];
    private static int _half, _positions;

    /// <summary>Tables covering at least <paramref name="positions"/> positions.</summary>
    public static (float[] Cos, float[] Sin) Tables(int positions, int headDim)
    {
        int half = headDim / 2;
        lock (Gate)
        {
            if (_half == half && _positions >= positions) return (_cos, _sin);
            int n = Math.Max(positions, Math.Max(_positions * 2, 1024));
            var inv = new float[half];
            for (int i = 0; i < half; i++) inv[i] = (float)Math.Pow(10000.0, -(float)i / half);
            var cos = new float[(long)n * half];
            var sin = new float[(long)n * half];
            for (int p = 0; p < n; p++)
                for (int i = 0; i < half; i++)
                {
                    float angle = p * inv[i];
                    cos[(long)p * half + i] = (float)Math.Cos(angle);
                    sin[(long)p * half + i] = (float)Math.Sin(angle);
                }
            (_cos, _sin, _half, _positions) = (cos, sin, half, n);
            return (cos, sin);
        }
    }

    /// <summary>Rotates one head vector <paramref name="x"/> (headDim floats) in place to <paramref name="position"/>, or
    /// back from it when <paramref name="inverse"/>: <c>(x1·c - x2·s, x1·s + x2·c)</c> over the two halves.</summary>
    public static void Apply(float* x, int headDim, float[] cos, float[] sin, int position, bool inverse = false)
    {
        int half = headDim / 2;
        fixed (float* cp = &cos[(long)position * half], sp = &sin[(long)position * half])
        {
            float* x1 = x, x2 = x + half;
            int i = 0;
            if (Vector512.IsHardwareAccelerated && !inverse)
            {
                for (; i + 16 <= half; i += 16)
                {
                    var a = Vector512.Load(x1 + i);
                    var b = Vector512.Load(x2 + i);
                    var c = Vector512.Load(cp + i);
                    var s = Vector512.Load(sp + i);
                    Vector512.Store(a * c - b * s, x1 + i);
                    Vector512.Store(a * s + b * c, x2 + i);
                }
            }
            for (; i < half; i++)
            {
                float a = x1[i], b = x2[i], c = cp[i], s = inverse ? -sp[i] : sp[i];
                x1[i] = a * c - b * s;
                x2[i] = a * s + b * c;
            }
        }
    }
}
