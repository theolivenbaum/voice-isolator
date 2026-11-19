using System.Text;
using System.Text.Json;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.IO;

/// <summary>One tensor of a <see cref="SafeTensorsFile"/>: its dtype, shape and byte range in the data section.</summary>
public sealed record SafeTensorInfo(string Name, string DType, int[] Shape, long Begin, long End)
{
    /// <summary>Number of elements.</summary>
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>
/// Reader for the <see href="https://github.com/huggingface/safetensors">safetensors</see> format: an 8-byte little-endian
/// header length, a JSON header naming every tensor's dtype, shape and byte range (plus a <c>__metadata__</c> map of
/// strings), then the raw little-endian data. The bytes come from a memory-mapped file or an in-memory buffer (a
/// decompressed <c>.vimodel</c>); tensors are read in place, without copies.
/// </summary>
public sealed unsafe class SafeTensorsFile : IDisposable
{
    private readonly MappedFile? _mapped;
    private readonly NativeBuffer<byte>? _buffer;
    private readonly byte* _data;
    private readonly long _length;
    private readonly Dictionary<string, SafeTensorInfo> _tensors;

    private SafeTensorsFile(string source, MappedFile? mapped, NativeBuffer<byte>? buffer, byte* bytes, long length)
    {
        Source = source;
        _mapped = mapped;
        _buffer = buffer;
        _length = length;
        if (length < 8) throw new InvalidDataException($"{source}: too short for a safetensors file.");
        long headerLength = *(long*)bytes;
        if (headerLength <= 0 || 8 + headerLength > length || headerLength > 100 << 20)
            throw new InvalidDataException($"{source}: not a safetensors file (header length {headerLength}).");
        string json = Encoding.UTF8.GetString(bytes + 8, (int)headerLength);
        _data = bytes + 8 + headerLength;
        long dataLength = length - 8 - headerLength;
        _tensors = new Dictionary<string, SafeTensorInfo>(StringComparer.Ordinal);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        using var doc = JsonDocument.Parse(json);
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__")
            {
                foreach (var m in property.Value.EnumerateObject()) metadata[m.Name] = m.Value.GetString() ?? "";
                continue;
            }
            var value = property.Value;
            var offsets = value.GetProperty("data_offsets");
            var info = new SafeTensorInfo(property.Name, value.GetProperty("dtype").GetString() ?? "",
                                          value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                                          offsets[0].GetInt64(), offsets[1].GetInt64());
            if (info.Begin < 0 || info.End < info.Begin || info.End > dataLength)
                throw new InvalidDataException($"{source}: tensor '{info.Name}' lies outside the file.");
            _tensors[info.Name] = info;
        }
        Metadata = metadata;
    }

    /// <summary>Where the bytes came from (a path or a resource name), for error messages.</summary>
    public string Source { get; }

    /// <summary>The <c>__metadata__</c> map of the header.</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors => _tensors;

    /// <summary>Memory-maps a <c>.safetensors</c> file.</summary>
    public static SafeTensorsFile Open(string path)
    {
        var mapped = new MappedFile(path);
        try
        {
            return new SafeTensorsFile(path, mapped, null, mapped.Ptr, mapped.Length);
        }
        catch
        {
            mapped.Dispose();
            throw;
        }
    }

    /// <summary>Takes ownership of a buffer holding a whole safetensors file.</summary>
    public static SafeTensorsFile FromBuffer(NativeBuffer<byte> buffer, string source)
    {
        try
        {
            return new SafeTensorsFile(source, null, buffer, buffer.Ptr, buffer.Length);
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <summary>A float32 tensor's data, checking its shape when <paramref name="shape"/> is given.</summary>
    public float* Float(string name, params int[] shape)
    {
        if (!_tensors.TryGetValue(name, out var info)) throw new InvalidDataException($"{Source}: missing tensor '{name}'.");
        if (info.DType != "F32") throw new InvalidDataException($"{Source}: tensor '{name}' is {info.DType}, expected F32.");
        if (info.End - info.Begin != info.Count * sizeof(float)) throw new InvalidDataException($"{Source}: tensor '{name}' has a bad size.");
        if (shape.Length > 0 && !shape.SequenceEqual(info.Shape))
            throw new InvalidDataException($"{Source}: tensor '{name}' has shape [{string.Join(", ", info.Shape)}], expected [{string.Join(", ", shape)}].");
        return (float*)(_data + info.Begin);
    }

    /// <summary>A float32 tensor copied into a managed array.</summary>
    public float[] FloatArray(string name, params int[] shape)
    {
        float* p = Float(name, shape);
        return new ReadOnlySpan<float>(p, checked((int)_tensors[name].Count)).ToArray();
    }

    public void Dispose()
    {
        _mapped?.Dispose();
        _buffer?.Dispose();
    }
}
