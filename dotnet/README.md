# Voice Isolator for .NET (pure C#)

A pure C# (.NET 11) port of the [Inworld Voice Isolator](../README.md) v1: 48 kHz voice isolation for noisy recordings
(`voice-isolator-v1`, offline) and live audio (`voice-isolator-v1-realtime`, a causal stream with 20 ms of look-ahead),
as a library, a command-line tool and two model packages.

- **No native dependencies**: no Python, no PyTorch, no ONNX Runtime, no BLAS. The STFT, the band encoders, the rotary
  time attention, the band attention, the bidirectional Mamba blocks with their selective scan, the decoder heads, the
  iSTFT, the streaming session and the resampler are all managed code (`System.Runtime.Intrinsics` `Vector512`,
  `Span<T>`, `System.Numerics.Tensors.TensorPrimitives`, a Goto-style AVX-512/AVX2 GEMM).
- **Loads the released checkpoints** as they are: the `.safetensors` files on
  [Hugging Face](https://huggingface.co/inworld/voice-isolator), memory-mapped, with the architecture read from their
  metadata and the release SHA-256 checked.
- **Validated layer by layer against the PyTorch model**: every stage of both checkpoints (STFT, encoders, each of the 7
  blocks and their halves, decoder, correction, output) matches to a relative error of at most 5e-6, the package's own
  stream matches for every push pattern, and the repository's **four golden fixtures pass with a maximum error of
  2.7e-7** against their 1e-4 tolerance. See [Validation](#validation).
- **4.4× faster than the Python package on the same CPU** (float32, 4 cores): an 8 s chunk takes 11.4 s instead of
  50 s. See [Performance](#performance).
- **The weights ship as NuGet packages**: each checkpoint, packed losslessly to fit nuget.org's 250 MB limit, is an
  embedded resource of its own package and is restored byte for byte (and checksum-verified) when loaded.

## Quick start

### Command line

```bash
cd dotnet
# offline (downloads voice-isolator-v1 to ~/.cache/voice-isolator on first use, 256 MB):
# the voice goes to sample_14_clean.wav and the removed background to sample_14_clean_background.wav
dotnet run -c Release --project src/VoiceIsolator.Cli -- ../demo/examples/sample_14.wav sample_14_clean.wav

# streaming, as a live session runs it (voice-isolator-v1-realtime, 20 ms of look-ahead)
dotnet run -c Release --project src/VoiceIsolator.Cli -- ../demo/examples/sample_14.wav sample_14_streamed.wav --mode streaming --lookahead 2

# a directory of WAVs, a local checkpoint, the input's own rate
dotnet run -c Release --project src/VoiceIsolator.Cli -- in/ out/ --model ../models/denoiser.safetensors --output-sr 0
```

The options are the Python CLI's (`--mode`, `--lookahead`, `--block-frames`, `--window-frames`, `--crossfade-frames`,
`--chunk-seconds`, `--overlap-seconds`, `--output-sr`, `--background`, `--no-background`, `--float32`), plus
`--threads`. Offline writes float32 WAVs at 48 kHz whose voice and background together reconstruct the input; streaming
writes the voice only. Other commands:

| Command | Does |
|---|---|
| `voice-isolate download --model voice-isolator-v1-realtime [--out path]` | fetches a released checkpoint (pinned revision, size and SHA-256 checked) and prints its path |
| `voice-isolate pack --model voice-isolator-v1 --out x.vimodel` | packs a checkpoint for a model package (see [Model packages](#model-packages)) |
| `voice-isolate info [--model …]` | prints a checkpoint's architecture, look-ahead choices and latency |
| `voice-isolate bench [--model …] [--seconds 8] [--runs 3] [--threads n]` | times repeated isolations; `VOICE_ISOLATOR_PROFILE=1` adds a per-stage breakdown |

### Library

```bash
dotnet add package VoiceIsolatorSharp                    # namespace VoiceIsolation
dotnet add package VoiceIsolatorSharp.Models.Offline     # optional: the voice-isolator-v1 weights, embedded
dotnet add package VoiceIsolatorSharp.Models.Realtime    # optional: the voice-isolator-v1-realtime weights, embedded
```

```csharp
using VoiceIsolation;
using VoiceIsolation.Audio;

// Offline: recorded audio, best quality.
using var isolator = await VoiceIsolator.LoadAsync(VoiceIsolatorModel.Offline);
var (wav, rate) = WaveFile.Read("noisy.wav");             // any rate, mixed to mono
float[] voice = isolator.Isolate(wav, rate);              // at the input rate, the input's length
// the removed background is wav[i] - voice[i]
WaveFile.WriteMonoFloat("voice.wav", voice, rate);

// Streaming: 48 kHz mono in pieces of any size, the samples that became final out.
using var live = await VoiceIsolator.LoadAsync(VoiceIsolatorModel.Realtime,
                                               new VoiceIsolatorOptions { LookaheadFrames = 2 });
using var session = live.CreateStream();
foreach (float[] piece in microphone)                     // e.g. 20 ms (960 samples) at a time
    Play(session.Push(piece));                            // latency: live.LatencySamples (3008 = 63 ms)
Play(session.Flush());                                    // the rest, once the source ends
```

`LoadAsync` takes the weights from the model package when it is referenced, else from the cache
(`$VOICE_ISOLATOR_CACHE`, default `~/.cache/voice-isolator`), else downloads them once from the pinned Hugging Face
revision; every source is checked against the release SHA-256. `VoiceIsolator.Load(path)` takes a local `.safetensors`
or `.vimodel`, and `OfflineModel.Load()` / `RealtimeModel.Load()` (namespace `VoiceIsolation.Models`) load a model
package explicitly. `VoiceIsolatorOptions` sets `Threads`, `LookaheadFrames`, `BlockFrames`, `CacheFrames`,
`WindowFrames`, `CrossfadeFrames`, `ChunkSeconds`, `OverlapSeconds` and `OfflinePasses`, with the Python package's
defaults. One `VoiceIsolator` is thread-safe and serves concurrent `Isolate` calls; a stream is one session.

The stages are public too: `VoiceIsolatorNetwork` (`Forward` over one chunk with a per-stage `TraceCallback`,
`Enhance` over any length, `SetLookahead`), `VoiceIsolatorStream`, `SafeTensorsFile`, `ModelArchive`, `WaveFile` and
`Resampler` (torchaudio's `kaiser_best`, the Python package's resampler).

## Packages

| Package | Contents | Size |
|---|---|---|
| `VoiceIsolatorSharp` | the library (namespace `VoiceIsolation`) | 0.1 MB |
| `VoiceIsolatorSharp.Models.Offline` | `voice-isolator-v1` (64.0M parameters), packed | 218 MB |
| `VoiceIsolatorSharp.Models.Realtime` | `voice-isolator-v1-realtime` (58.2M parameters), packed | 198 MB |
| `VoiceIsolatorSharp.Cli` | the `voice-isolate` .NET tool | 0.3 MB |

The two checkpoints are 256 MB and 233 MB of float32 weights: together they cannot fit one package under nuget.org's
250 MB limit, and plain deflate (what a `.nupkg` applies) saves only 7%, so each model has its own package with the
weights in the `.vimodel` container described below. The model packages depend on `VoiceIsolatorSharp`; referencing one
is enough. (A trimmed or single-file publish must keep the model assembly, which the library loads by name.)

## How it works

```
48 kHz mono ─► STFT (n_fft 2048, hop 480, periodic Hann, reflect-padded)                 [frames, 1025 bins] complex
           ─► 46 bands (100 Hz wide up to 1 kHz … 2 kHz wide up to 24 kHz), per band: LayerNorm ─► Linear(2·bins → 512)
              + band embedding (+ look-ahead embedding)                                   [46 bands, frames, 512]
           ─► 7 dual-path blocks, alternating:
                attention: time  x += Wo·XSA(RoPE MHA(LN x)) ; x += FFN(LN x)        (8 heads, causal + 1200-frame window
                           bands x += Wo·MHA(LN x) ; x += FFN(LN x)                    in the realtime model)
                Mamba:     time  x += LN(Wo·[fwd(x) + x, bwd(x) + x])              (state 16, conv 4, expand 2;
                           bands x += LN(Wo·[fwd(x) + x, bwd(x) + x])               the realtime time path is forward only)
           ─► per band: FiLM ─► LayerNorm ─► Linear ─► GELU ─► Linear(512 → 4·bins)   [frames, bins, 4]
              (read 0/2/4 frames ahead in the realtime model)
           ─► correction: S·(2 tanh r₀ + 2i tanh r₁) + (tanh r₂, tanh r₃)/max(1, ‖·‖)·0.5·|S|
           ─► iSTFT ─► + input                                                         isolated voice
```

XSA is the exclusive self attention of the model card: each token's attention output loses its component along its own
value. Long offline inputs render in 8 s windows overlapping by 2 s with a linear crossfade, exactly as the Python
package places them. The architecture is described in the [model card](../MODEL_CARD.md#architecture).

| Source | Contents |
|---|---|
| `src/VoiceIsolator/VoiceIsolator.cs` | Public API: load (embedded, cached or downloaded), `Isolate`, `CreateStream`, options |
| `src/VoiceIsolator/VoiceIsolatorModels.cs` | Release catalog: names, Hub revision, SHA-256, sizes, model packages |
| `src/VoiceIsolator/Model/VoiceIsolatorNetwork.cs` | The network over a chunk (`Forward`, traced), chunked `Enhance`, the dual-path blocks |
| `src/VoiceIsolator/Model/AttentionBlocks.cs` | Time transformer (RoPE, causal window, exclusive projection; GEMM-blocked attention) and band transformer |
| `src/VoiceIsolator/Model/Mamba.cs` | Mamba layer (projections, depthwise conv, softplus, AVX-512 selective scan), bidirectional wrapper, stream state |
| `src/VoiceIsolator/Model/BandCodec.cs` | Band encoders and decoder heads |
| `src/VoiceIsolator/Model/Spectral.cs` | STFT, iSTFT and the bounded correction |
| `src/VoiceIsolator/Model/VoiceIsolatorStream.cs` | Streaming session: rotary key cache with compaction, carried Mamba state, two-lane windows, overlap-add release |
| `src/VoiceIsolator/Model/VoiceIsolatorConfig.cs` | Architecture parsed from the checkpoint (unknown fields refused), band edges |
| `src/VoiceIsolator/Numerics/` | GEMM with fused bias/GELU/residual epilogues, packing, worker pool, vector `exp`/`erf`/`log`, FFT, LayerNorm, softmax |
| `src/VoiceIsolator/IO/` | Safetensors reader (memory-mapped), `.vimodel` container, resumable downloader |
| `src/VoiceIsolator/Audio/` | WAV reading/writing, `kaiser_best` resampler |
| `src/VoiceIsolator.Models.*` | The model packages (`ModelPackage.targets` embeds the packed weights) |
| `src/VoiceIsolator.Cli/` | `voice-isolate` |
| `tests/VoiceIsolator.Tests/` | Kernel tests, layer-by-layer parity, stream parity, golden fixtures |
| `scripts/dump_reference.py` | PyTorch reference dumps for the parity tests |
| `scripts/gen_kernels.py` | Generator of the unrolled AVX-512 GEMM micro-kernels |

### Numerics

Everything runs in float32 like the reference's `fp32` precision (the Python package runs bfloat16 only under CUDA).
Weights are packed once at load into the GEMM's panel layout. Biases, GELU and residual additions are fused into the GEMM
epilogues. The time attention packs each head's keys and values as GEMM operands and runs blocks of 64 queries through
score GEMM, mask, softmax and value GEMM while they are cache-hot; causal blocks skip the panels past their last query.
The selective scan keeps 16 channels' states in vector registers and evaluates the 16 `exp(Δ·A)` per frame with a
Cephes-style vector exponential (≈1 ulp). GELU uses the exact-GELU formula with the same `erf` approximation as
PyTorch's vectorized CPU kernel; softplus uses a compensated `log1p`. The FFT runs in double precision.

## Validation

The tests compare the port with the PyTorch model in three ways (`dotnet test tests/VoiceIsolator.Tests -c Release`).
The model tests need the checkpoints (`VOICE_ISOLATOR_MODELS_DIR`, default the repository's `models/`) and, for the
layer tests, the reference dumps (`VOICE_ISOLATOR_REFERENCE_DIR`, default `staging/dotnet-ref`); without them they print
`SKIPPED` and pass.

**Layer by layer** (`ParityTests.EveryLayerMatchesPyTorch`), on the 3 s fixture clip, against forward hooks on the
PyTorch network (relative L2 error of the whole tensor):

| Stage | voice-isolator-v1 | voice-isolator-v1-realtime |
|---|---|---|
| STFT | 1.6e-7 | 1.6e-7 |
| band encoders | 3.4e-7 | 3.4e-7 |
| block 0 (attention): time attention / time half / block | 1.2e-6 / 1.1e-6 / 1.4e-6 | 7.2e-7 / 1.1e-6 / 1.3e-6 |
| block 1 (Mamba): forward scan / time / bands / block | 1.4e-6 / 9.3e-7 / 8.5e-7 / 8.6e-7 | 1.4e-6 / 8.1e-7 / 1.0e-6 / 8.9e-7 |
| block 3 (Mamba) | 2.0e-6 | 2.0e-6 |
| block 6 (attention, last) | 1.2e-6 | 1.2e-6 |
| decoder heads | 5.8e-7 | 6.8e-7 |
| correction | 1.2e-6 | 1.2e-6 |
| output audio | 4.6e-7 (max abs 3.6e-7) | 3.4e-7 (max abs 2.1e-7) |

No stage exceeds 4.6e-6. **Streaming** (`StreamMatchesPyTorchStream`): the C# session against the Python package's
`Stream` over the 3 s and 11 s clips, plain and windowed (the 11 s clip crosses two lane crossfades), pushed 960 and 7001
samples at a time: maximum error 3.6e-7. **Golden fixtures** (`GoldenTests`): the repository's
[`tests/fixtures`](../tests/fixtures/README.md), rendered by the independent training reference, at their own settings
(offline in 8 s chunks; streaming in 0.5 s pushes with 600-frame windows):

| Fixture | Max abs error (tolerance 1e-4) |
|---|---|
| `offline_3s` | 1.8e-7 |
| `offline_11s` | 2.4e-7 |
| `realtime_stream_3s` | 2.0e-7 |
| `realtime_stream_11s` | 2.7e-7 |

To regenerate the reference dumps (needs the Python package, `pip install -e .` from the repository root):

```bash
python dotnet/scripts/dump_reference.py --model models/denoiser.safetensors \
    --input tests/fixtures/offline_3s/input.wav --out staging/dotnet-ref/offline
python dotnet/scripts/dump_reference.py --model models/denoiser_realtime.safetensors --lookahead 2 \
    --input tests/fixtures/offline_3s/input.wav --out staging/dotnet-ref/realtime
python dotnet/scripts/dump_reference.py --model models/denoiser_realtime.safetensors --lookahead 2 \
    --input tests/fixtures/offline_11s/input.wav --out staging/dotnet-ref/realtime_11s --streams-only
```

The kernel tests (GEMM with every epilogue against a double-precision product, FFT against a DFT, `erf`, softplus,
the `.vimodel` round trip, the resampler, `linspace`, the band layout) need nothing and always run. The API tests check
rates and lengths, concurrent calls against sequential ones, that a windowed stream's output does not depend on how the
audio is pushed, and that `Isolate` in streaming mode is the half-second stream. The fallback paths pass the same
tests: AVX2 only (`DOTNET_EnableAVX512=0`: every layer and golden fixture, at the same errors) and scalar
(`DOTNET_EnableAVX2=0`: the kernel tests).

## Performance

`voice-isolate bench`, 8 s of audio, median of 3 runs, float32, on a 4-core Intel Xeon (AVX-512) at 2.1 GHz:

| | 4 threads | 1 thread |
|---|---|---|
| `voice-isolator-v1`, one 8 s chunk | 11.4 s (RTF 1.42) | 40.9 s (RTF 5.1) |
| `voice-isolator-v1-realtime`, stream (4-frame blocks, 600/200-frame windows) | 17.7 s (RTF 2.2) | |
| Python package, `voice-isolator-v1`, one 8 s chunk, CPU, fp32 (PyTorch 2.15, 4 threads) | 49.8 s (RTF 6.2) | |

The offline pass spends half its time in the attention blocks and half in the Mamba blocks, almost all of it in GEMMs
(~175 GFLOP/s); the selective scan is about a tenth. A stream computes 4-frame blocks (184 tokens per GEMM) and, inside
crossfades, two lanes. Neither is real time on four cores; a machine with more cores scales nearly linearly (3.6× from 1
to 4 threads here), and `BlockFrames` raises a stream's throughput without adding latency.

## Model packages

`voice-isolate pack` writes the `.vimodel` container (`IO/ModelArchive.cs`):

```
"VIMODEL1" | u64 original length | SHA-256 of the original | u32 element size (4) | u32 prefix length | zlib stream
```

The zlib stream holds the safetensors header as it is, then the float data byte-shuffled: every weight's first bytes,
then its second bytes, and so on. The sign-and-exponent plane compresses to ~40%, the file to 85% (256 → 218 MB,
233 → 198 MB). Unpacking restores the original file byte for byte and checks its SHA-256 against the container's and the
release's. Building a model package embeds `artifacts/models/<name>.vimodel` (or `$(VoiceIsolatorModelsDir)`); without
it the project still builds (so the solution builds anywhere) and `dotnet pack` fails instead of publishing an empty
model:

```bash
dotnet run -c Release --project src/VoiceIsolator.Cli -- pack --model voice-isolator-v1 --out artifacts/models/voice-isolator-v1.vimodel
dotnet run -c Release --project src/VoiceIsolator.Cli -- pack --model voice-isolator-v1-realtime --out artifacts/models/voice-isolator-v1-realtime.vimodel
dotnet pack src/VoiceIsolator.Models.Offline -c Release -o artifacts/packages
```

## Build, test and CI

```bash
cd dotnet
dotnet build VoiceIsolator.slnx -c Release
dotnet test tests/VoiceIsolator.Tests -c Release
```

The .NET 11 SDK is required. `.devops/azure-pipelines.yml` builds, tests, packs the weights and the four packages with a
CalVer version (`yy.M.<buildId>`) and pushes them on `main`, like the ParakeetSharp pipeline it follows.

## Scope and differences from the Python package

- Float32 only: no bfloat16 or float16 (the Python package uses them on CUDA only).
- CPU only. The vector kernels use AVX-512 where available; the GEMM also has an AVX2 path, and everything else falls
  back to scalar code, so ARM64 runs correctly but without SIMD tuning.
- A stream is one clip. The Python package can stream padded batches of clips in lockstep; run several sessions instead.
- Checkpoints with `dual_mode` or `mode_film` are refused (the released models use neither), as are unknown config fields.
- Resampling (`kaiser_best`) computes its kernel in double precision; torchaudio computes it in float32.

## Licenses

The port is Apache-2.0 like the rest of this repository. The GEMM, packing, worker pool, vector math, LayerNorm and
softmax kernels, the WAV reader, the memory-mapped file and the downloader are adapted from
[curiosity-ai/parakeet](https://github.com/curiosity-ai/parakeet) (Apache-2.0). The weights are Inworld's
(Apache-2.0, see the [model card](../MODEL_CARD.md) and [NOTICE](../NOTICE)). The golden fixtures used by the tests are
CC BY 4.0 (see [tests/fixtures](../tests/fixtures/README.md)) and are not part of any package.
