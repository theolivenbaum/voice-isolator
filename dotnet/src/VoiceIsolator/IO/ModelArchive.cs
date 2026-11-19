using System.IO.Compression;
using System.Security.Cryptography;
using VoiceIsolation.Numerics;

namespace VoiceIsolation.IO;

/// <summary>
/// The <c>.vimodel</c> container: a checkpoint compressed losslessly for the model NuGet packages. Float weights barely
/// compress as they are (deflate saves ~7%), but split into byte planes (every element's first bytes, then its second
/// bytes, ...) the sign-and-exponent plane compresses to about 40%, and the file to ~85%: the released offline model
/// (256 MB) packs to ~218 MB and the realtime model (233 MB) to ~198 MB, each under nuget.org's 250 MB package limit.
/// <code>
/// "VIMODEL1" | u64 original length | 32-byte SHA-256 of the original | u32 element size | u32 prefix length | zlib stream
/// </code>
/// The zlib stream holds the first <c>prefix</c> bytes (the safetensors header) as they are, then the rest of the file
/// byte-shuffled in elements of <c>element size</c> bytes (a tail shorter than one element stays as it is). Unpacking
/// restores the original file byte for byte and checks its SHA-256, so a packed model is exactly the released one.
/// </summary>
public static unsafe class ModelArchive
{
    private static ReadOnlySpan<byte> Magic => "VIMODEL1"u8;
    private const int HeaderSize = 8 + 8 + 32 + 4 + 4;

    /// <summary>File extension of a packed model.</summary>
    public const string Extension = ".vimodel";

    /// <summary>Whether <paramref name="header"/> (the first bytes of a file) starts a packed model.</summary>
    public static bool IsArchive(ReadOnlySpan<byte> header) => header.Length >= Magic.Length && header[..Magic.Length].SequenceEqual(Magic);

    /// <summary>Packs a safetensors checkpoint into <paramref name="destination"/>.</summary>
    public static void Pack(string safetensorsPath, Stream destination, CompressionLevel level = CompressionLevel.SmallestSize)
    {
        byte[] original = File.ReadAllBytes(safetensorsPath);
        long headerLength = BitConverter.ToInt64(original, 0);
        if (headerLength <= 0 || 8 + headerLength > original.Length) throw new InvalidDataException($"{safetensorsPath}: not a safetensors file.");
        int prefix = checked((int)(8 + headerLength));
        const int element = sizeof(float);

        Span<byte> header = stackalloc byte[HeaderSize];
        Magic.CopyTo(header);
        BitConverter.TryWriteBytes(header[8..], (ulong)original.Length);
        SHA256.HashData(original, header[16..48]);
        BitConverter.TryWriteBytes(header[48..], (uint)element);
        BitConverter.TryWriteBytes(header[52..], (uint)prefix);
        destination.Write(header);

        byte[] shuffled = new byte[original.Length];
        original.AsSpan(0, prefix).CopyTo(shuffled);
        Shuffle(original.AsSpan(prefix), shuffled.AsSpan(prefix), element);
        using var z = new ZLibStream(destination, level, leaveOpen: true);
        z.Write(shuffled);
    }

    /// <summary>Unpacks a packed model into a buffer holding the original file; checks its SHA-256 (and that it equals
    /// <paramref name="expectedSha256"/> when given).</summary>
    public static NativeBuffer<byte> Unpack(Stream source, string? expectedSha256 = null, string name = "model")
    {
        Span<byte> header = stackalloc byte[HeaderSize];
        source.ReadExactly(header);
        if (!IsArchive(header)) throw new InvalidDataException($"{name}: not a {Extension} file.");
        long length = (long)BitConverter.ToUInt64(header[8..]);
        byte[] digest = header[16..48].ToArray();
        int element = (int)BitConverter.ToUInt32(header[48..]);
        int prefix = (int)BitConverter.ToUInt32(header[52..]);
        if (length <= 0 || length > int.MaxValue || prefix > length || element is < 1 or > 16)
            throw new InvalidDataException($"{name}: corrupt {Extension} header.");
        if (expectedSha256 != null && !Convert.ToHexStringLower(digest).Equals(expectedSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"{name}: packs a checkpoint with SHA-256 {Convert.ToHexStringLower(digest)}, expected {expectedSha256}.");

        var output = new NativeBuffer<byte>(length, zero: false, hugePages: false);
        try
        {
            using (var shuffled = new NativeBuffer<byte>(length, zero: false, hugePages: false))
            {
                using (var z = new ZLibStream(source, CompressionMode.Decompress, leaveOpen: true))
                {
                    var span = new Span<byte>(shuffled.Ptr, (int)length);
                    z.ReadExactly(span);
                }
                var all = new ReadOnlySpan<byte>(shuffled.Ptr, (int)length);
                var dst = new Span<byte>(output.Ptr, (int)length);
                all[..prefix].CopyTo(dst);
                Unshuffle(all[prefix..], dst[prefix..], element);
            }
            Span<byte> actual = stackalloc byte[32];
            SHA256.HashData(new ReadOnlySpan<byte>(output.Ptr, (int)length), actual);
            if (!actual.SequenceEqual(digest)) throw new InvalidDataException($"{name}: checksum mismatch after unpacking.");
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    /// <summary>Splits <paramref name="src"/> into byte planes of <paramref name="element"/>-byte elements.</summary>
    public static void Shuffle(ReadOnlySpan<byte> src, Span<byte> dst, int element)
    {
        int count = src.Length / element;
        for (int b = 0; b < element; b++)
        {
            var plane = dst.Slice(b * count, count);
            for (int i = 0; i < count; i++) plane[i] = src[i * element + b];
        }
        src[(count * element)..].CopyTo(dst[(count * element)..]);
    }

    /// <summary>Inverse of <see cref="Shuffle"/>.</summary>
    public static void Unshuffle(ReadOnlySpan<byte> src, Span<byte> dst, int element)
    {
        int count = src.Length / element;
        if (element == 4)
        {
            // Interleave four planes a 32-bit element at a time.
            fixed (byte* s = src)
            fixed (byte* d = dst)
            {
                byte* p0 = s, p1 = s + count, p2 = s + 2L * count, p3 = s + 3L * count;
                uint* o = (uint*)d;
                for (int i = 0; i < count; i++) o[i] = p0[i] | (uint)p1[i] << 8 | (uint)p2[i] << 16 | (uint)p3[i] << 24;
            }
        }
        else
        {
            for (int b = 0; b < element; b++)
            {
                var plane = src.Slice(b * count, count);
                for (int i = 0; i < count; i++) dst[i * element + b] = plane[i];
            }
        }
        src[(count * element)..].CopyTo(dst[(count * element)..]);
    }
}
