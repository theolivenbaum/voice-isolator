using System.Text.Json;

namespace VoiceIsolation.Model;

/// <summary>
/// Architecture of a checkpoint (the Python package's <c>VoiceIsolatorConfig</c>), always read from the checkpoint's
/// <c>config</c> metadata. Training-only fields are ignored; a value this port does not implement is refused rather than
/// guessed.
/// </summary>
public sealed record VoiceIsolatorConfig
{
    public int Width { get; init; } = 320;
    public string[] BlockTypes { get; init; } = ["transformer", "mamba", "transformer", "mamba", "transformer", "mamba", "transformer"];
    public int NumHeads { get; init; } = 8;
    public int NFft { get; init; } = 2048;
    public int HopLength { get; init; } = 480;
    public int SampleRate { get; init; } = 48000;
    public int[][] BandPlan { get; init; } = [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]];
    public float CorrectionBound { get; init; } = 0.5f;
    public double ChunkSeconds { get; init; } = 12.0;
    public double OverlapSeconds { get; init; } = 2.0;
    public bool ExclusiveSelfAttention { get; init; } = true;
    public string ExclusiveAxes { get; init; } = "time";
    public int MambaState { get; init; } = 16;
    public int MambaConv { get; init; } = 4;
    public int MambaExpand { get; init; } = 2;
    public bool Causal { get; init; }
    public int LookaheadFrames { get; init; }
    public int[] LookaheadFramesChoices { get; init; } = [];
    public bool DualMode { get; init; }
    public bool ModeFilm { get; init; }
    public int AttentionWindowFrames { get; init; }
    public int OfflinePasses { get; init; } = 1;

    public int HeadDim => Width / NumHeads;
    public int MambaInner => MambaExpand * Width;
    public int DtRank => (Width + 15) / 16;
    public int Bins => NFft / 2 + 1;
    public bool ExclusiveOnTime => ExclusiveSelfAttention && ExclusiveAxes is "both" or "time";
    public bool ExclusiveOnBand => ExclusiveSelfAttention && ExclusiveAxes is "both" or "band";

    private static readonly HashSet<string> TrainingOnly =
    [
        "activation_checkpointing", "checkpoint_every", "attention_backend", "offline_mode_prob", "dual_distill", "hint_sha256",
        "hint_bf16", "grid_width", "grid_hidden", "grid_kernel", "grid_stride", "grid_heads", "grid_qk_channels", "grid_output",
        "grid_normalize_input", "decision_head",
    ];

    private static readonly Dictionary<string, string?> Required = new()
    {
        ["representation"] = "bands",
        ["decoder_type"] = "complex_mask_residual",
        ["hint_checkpoint"] = null,
    };

    /// <summary>Parses the checkpoint's architecture JSON (snake_case keys, as the Python package writes them).</summary>
    public static VoiceIsolatorConfig Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("width", out _))
            throw new InvalidDataException("No embedded architecture config; refusing to guess the architecture.");
        var c = new VoiceIsolatorConfig();
        var unknown = new List<string>();
        foreach (var p in root.EnumerateObject())
        {
            var v = p.Value;
            switch (p.Name)
            {
                case "width": c = c with { Width = v.GetInt32() }; break;
                case "block_types": c = c with { BlockTypes = v.EnumerateArray().Select(e => e.GetString()!).ToArray() }; break;
                case "num_heads": c = c with { NumHeads = v.GetInt32() }; break;
                case "n_fft": c = c with { NFft = v.GetInt32() }; break;
                case "hop_length": c = c with { HopLength = v.GetInt32() }; break;
                case "sample_rate": c = c with { SampleRate = v.GetInt32() }; break;
                case "band_plan": c = c with { BandPlan = v.EnumerateArray().Select(e => e.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray() }; break;
                case "correction_bound": c = c with { CorrectionBound = v.GetSingle() }; break;
                case "chunk_seconds": c = c with { ChunkSeconds = v.GetDouble() }; break;
                case "overlap_seconds": c = c with { OverlapSeconds = v.GetDouble() }; break;
                case "exclusive_self_attention": c = c with { ExclusiveSelfAttention = v.GetBoolean() }; break;
                case "exclusive_axes": c = c with { ExclusiveAxes = v.GetString()! }; break;
                case "mamba_state": c = c with { MambaState = v.GetInt32() }; break;
                case "mamba_conv": c = c with { MambaConv = v.GetInt32() }; break;
                case "mamba_expand": c = c with { MambaExpand = v.GetInt32() }; break;
                case "causal": c = c with { Causal = v.GetBoolean() }; break;
                case "lookahead_frames": c = c with { LookaheadFrames = v.GetInt32() }; break;
                case "lookahead_frames_choices": c = c with { LookaheadFramesChoices = v.EnumerateArray().Select(e => e.GetInt32()).ToArray() }; break;
                case "dual_mode": c = c with { DualMode = v.GetBoolean() }; break;
                case "mode_film": c = c with { ModeFilm = v.GetBoolean() }; break;
                case "attention_window_frames": c = c with { AttentionWindowFrames = v.GetInt32() }; break;
                case "offline_passes": c = c with { OfflinePasses = v.GetInt32() }; break;
                default:
                    if (Required.TryGetValue(p.Name, out var want))
                    {
                        string? got = v.ValueKind == JsonValueKind.Null ? null : v.ToString();
                        if (got != want) throw new NotSupportedException($"This port implements {p.Name}={want ?? "None"}, not {got}.");
                    }
                    else if (!TrainingOnly.Contains(p.Name) && !IsEmpty(v)) unknown.Add(p.Name);
                    break;
            }
        }
        if (unknown.Count > 0)
            throw new NotSupportedException($"Checkpoint config uses fields this port does not implement: [{string.Join(", ", unknown)}].");
        c.Validate();
        return c;
    }

    private static bool IsEmpty(JsonElement v) => v.ValueKind switch
    {
        JsonValueKind.Null or JsonValueKind.False => true,
        JsonValueKind.Number => v.GetDouble() == 0,
        JsonValueKind.String => v.GetString() == "",
        JsonValueKind.Array => v.GetArrayLength() == 0,
        JsonValueKind.Object => !v.EnumerateObject().Any(),
        _ => false,
    };

    /// <summary>Checks the same invariants as the Python config, plus what this port supports.</summary>
    public void Validate()
    {
        if (Width < 1 || NumHeads < 1 || Width % (2 * NumHeads) != 0) throw new InvalidDataException("width must be a positive multiple of twice num_heads.");
        if (BlockTypes.Length == 0 || BlockTypes.Any(b => b is not ("transformer" or "mamba"))) throw new InvalidDataException("block_types must be transformer or mamba.");
        if (NFft < 4 || NFft % 2 != 0 || HopLength <= 0 || HopLength > NFft / 2) throw new InvalidDataException("n_fft must be even and hop_length within (0, n_fft / 2].");
        if ((NFft & (NFft - 1)) != 0) throw new NotSupportedException("This port needs a power-of-two n_fft.");
        if (CorrectionBound is <= 0 or >= 1) throw new InvalidDataException("correction_bound must lie in (0, 1).");
        if (ExclusiveAxes is not ("both" or "time" or "band")) throw new InvalidDataException("exclusive_axes must be both, time or band.");
        if (Math.Min(MambaState, Math.Min(MambaConv, MambaExpand)) < 1) throw new InvalidDataException("Mamba dimensions must be positive.");
        if (LookaheadFrames < 0) throw new InvalidDataException("lookahead_frames cannot be negative.");
        if (LookaheadFramesChoices.Length > 0 && !LookaheadFramesChoices.Contains(LookaheadFrames))
            throw new InvalidDataException("lookahead_frames must be one of lookahead_frames_choices.");
        if ((LookaheadFrames > 0 || LookaheadFramesChoices.Length > 0) && !(Causal || DualMode))
            throw new InvalidDataException("A look-ahead needs a causal or dual-mode model.");
        if (AttentionWindowFrames < 0 || (AttentionWindowFrames > 0 && !Causal))
            throw new InvalidDataException("attention_window_frames needs a causal model and cannot be negative.");
        if (OfflinePasses < 1 || (OfflinePasses > 1 && Causal)) throw new InvalidDataException("offline_passes must be at least 1, and 1 for a causal model.");
        if (DualMode || ModeFilm)
            throw new NotSupportedException("Dual-mode and mode-FiLM checkpoints are not supported by this port (the released models use neither).");
        if (HeadDim % 16 != 0) throw new NotSupportedException("This port needs a head dimension that is a multiple of 16.");
        if (MambaInner % 16 != 0) throw new NotSupportedException("This port needs a Mamba inner width that is a multiple of 16.");
    }

    /// <summary>Bin ranges [start, end) of the bands, covering every bin once (<c>model.band_edges</c>).</summary>
    public (int Start, int End)[] BandEdges()
    {
        int bins = Bins;
        double hzPerBin = (double)SampleRate / NFft;
        var edges = new List<(int, int)>();
        int start = 0;
        foreach (var pair in BandPlan)
        {
            int width = Math.Max(1, (int)Math.Round(pair[0] / hzPerBin, MidpointRounding.ToEven));
            int upto = Math.Min(bins, (int)Math.Round(pair[1] / hzPerBin, MidpointRounding.ToEven));
            while (start < upto)
            {
                int end = Math.Min(start + width, upto);
                edges.Add((start, end));
                start = end;
            }
        }
        if (start < bins) edges.Add((start, bins));
        return edges.ToArray();
    }
}
