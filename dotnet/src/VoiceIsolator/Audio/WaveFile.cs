using System.Buffers.Binary;

namespace VoiceIsolation.Audio;

/// <summary>Minimal RIFF/WAVE reading and writing (adapted from curiosity-ai/parakeet, Apache-2.0): PCM 8/16/24/32-bit and
/// IEEE float 32/64-bit, any channel count (averaged to mono), and mono 16-bit PCM or 32-bit float output.</summary>
public static class WaveFile
{
    /// <summary>Reads a WAV file as mono float samples at its own rate.</summary>
    public static (float[] Samples, int SampleRate) Read(string path) => Decode(File.ReadAllBytes(path));

    /// <summary>Reads a WAV file as mono float samples at <paramref name="targetRate"/> Hz (kaiser_best resampling).</summary>
    public static float[] ReadMono(string path, int targetRate)
    {
        var (samples, rate) = Read(path);
        return Resampler.Resample(samples, rate, targetRate);
    }

    /// <summary>True when the bytes start with a RIFF/WAVE (or RF64) header.</summary>
    public static bool IsWave(ReadOnlySpan<byte> data) =>
        data.Length >= 12 && (data[..4].SequenceEqual("RIFF"u8) || data[..4].SequenceEqual("RF64"u8)) && data.Slice(8, 4).SequenceEqual("WAVE"u8);

    /// <summary>Decodes WAV bytes to mono float samples at their native rate.</summary>
    public static (float[] Samples, int SampleRate) Decode(ReadOnlySpan<byte> wav)
    {
        if (!IsWave(wav)) throw new InvalidDataException("Not a RIFF/WAVE file.");
        int format = 0, channels = 0, rate = 0, bits = 0;
        ReadOnlySpan<byte> data = default;
        bool haveData = false;
        int pos = 12;
        while (pos + 8 <= wav.Length)
        {
            var id = wav.Slice(pos, 4);
            long size = BinaryPrimitives.ReadUInt32LittleEndian(wav[(pos + 4)..]);
            int body = pos + 8;
            if (id.SequenceEqual("fmt "u8))
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(wav[body..]);
                channels = BinaryPrimitives.ReadUInt16LittleEndian(wav[(body + 2)..]);
                rate = BinaryPrimitives.ReadInt32LittleEndian(wav[(body + 4)..]);
                bits = BinaryPrimitives.ReadUInt16LittleEndian(wav[(body + 14)..]);
                if (format == 0xFFFE && size >= 26) format = BinaryPrimitives.ReadUInt16LittleEndian(wav[(body + 24)..]);
            }
            else if (id.SequenceEqual("data"u8))
            {
                // Streams written without a final size (0 or 0xFFFFFFFF) run to the end of the file.
                long avail = wav.Length - body;
                long len = size == 0 || size == 0xFFFFFFFF || size > avail ? avail : size;
                data = wav.Slice(body, (int)len);
                haveData = true;
                break;
            }
            pos = body + (int)Math.Min(size + (size & 1), wav.Length - body);
        }
        if (!haveData || channels <= 0 || rate <= 0) throw new InvalidDataException("WAV file has no fmt or data chunk.");
        int bytesPer = bits / 8;
        if (bytesPer <= 0) throw new InvalidDataException($"Unsupported WAV sample size {bits}.");
        int frames = data.Length / (bytesPer * channels);
        var mono = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                var s = data.Slice((f * channels + c) * bytesPer, bytesPer);
                sum += (format, bits) switch
                {
                    (1, 8) => (s[0] - 128) / 128f,
                    (1, 16) => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
                    (1, 24) => ((s[0] << 8 | s[1] << 16 | s[2] << 24) >> 8) / 8388608f,
                    (1, 32) => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
                    (3, 32) => BinaryPrimitives.ReadSingleLittleEndian(s),
                    (3, 64) => (float)BinaryPrimitives.ReadDoubleLittleEndian(s),
                    _ => throw new InvalidDataException($"Unsupported WAV encoding (format {format}, {bits} bits)."),
                };
            }
            mono[f] = sum / channels;
        }
        return (mono, rate);
    }

    /// <summary>Writes mono samples as 16-bit PCM WAV. Integer samples cannot hold values beyond full scale, so a
    /// waveform peaking above 1.0 is scaled down to peak at 1.0 instead of being clipped (as the Python package does).</summary>
    public static void WriteMono16(string path, ReadOnlySpan<float> samples, int sampleRate)
    {
        float peak = samples.IsEmpty ? 0f : System.Numerics.Tensors.TensorPrimitives.MaxMagnitude(samples);
        float gain = peak > 1f ? 1f / peak : 1f;
        var bytes = new byte[44 + samples.Length * 2];
        "RIFF"u8.CopyTo(bytes);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), 36 + samples.Length * 2);
        "WAVEfmt "u8.CopyTo(bytes.AsSpan(8));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16), 16);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(20), 1);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(22), 1);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28), sampleRate * 2);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(32), 2);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(34), 16);
        "data"u8.CopyTo(bytes.AsSpan(36));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40), samples.Length * 2);
        for (int i = 0; i < samples.Length; i++)
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(44 + 2 * i), (short)Math.Round(Math.Clamp(samples[i] * gain, -1f, 1f) * 32767f));
        File.WriteAllBytes(path, bytes);
    }

    /// <summary>Writes mono samples as 32-bit IEEE float WAV, unchanged.</summary>
    public static void WriteMonoFloat(string path, ReadOnlySpan<float> samples, int sampleRate)
    {
        var bytes = new byte[44 + samples.Length * 4];
        "RIFF"u8.CopyTo(bytes);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), 36 + samples.Length * 4);
        "WAVEfmt "u8.CopyTo(bytes.AsSpan(8));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16), 16);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(20), 3);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(22), 1);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28), sampleRate * 4);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(32), 4);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(34), 32);
        "data"u8.CopyTo(bytes.AsSpan(36));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40), samples.Length * 4);
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(samples).CopyTo(bytes.AsSpan(44));
        File.WriteAllBytes(path, bytes);
    }
}
