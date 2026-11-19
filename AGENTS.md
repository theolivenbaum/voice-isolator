# Inworld Voice Isolator — guide for coding agents

This file serves two tasks:

- **Setting up the Voice Isolator for someone to use** (install it, check the
  machine, clean a first file): follow [Setting up the Voice Isolator for a user](#setting-up-the-voice-isolator-for-a-user),
  the last section, and skip the rest. It installs with `pip`; `make install` is
  the contributor setup, with development and evaluation extras a user does not need.
- **Changing this repository:** read every section before that one first.
  [CONTRIBUTING.md](CONTRIBUTING.md) covers setup and validation;
  [README.md](README.md) covers usage.

`CLAUDE.md` imports this file, and `.claude/skills/` links to `.agents/skills/`.

## Repository layout

| Path | Purpose |
|---|---|
| `src/inworld/voice_isolation/` | Inference package; `__init__.py` defines the public API |
| `tests/` | Tests mirroring source paths; `tests/fixtures/` holds reference parity cases |
| `bench/scripts/`, `bench/wrappers/` | Benchmark tools and external-system adapters; see [bench/README.md](bench/README.md) |
| `demo/` | Interactive demo; see [demo/README.md](demo/README.md) |
| `examples/` | Guided notebook for a first evaluation; see [examples/README.md](examples/README.md) |
| `docs/` | Figure generator and README assets |
| `dotnet/` | Pure C# (.NET 11) port: library, CLI, model packages and parity tests; see [dotnet/README.md](dotnet/README.md) |
| `MODEL_CARD.md` | The model card, shown as the Hugging Face page: purpose, quick start, benchmark, architecture, the two checkpoints and training data |
| `models/`, `data/`, `staging/` | Gitignored checkpoints, local data and scratch files |

## Coding conventions

- Follow Google Python style, enforced by the pinned `isort`, `yapf` and `pylint`
  tools. Use Google-style docstrings, type hints on public signatures, and module
  imports rather than imports of individual functions or classes. Standard
  aliases such as `np`, `sf`, `plt` and `gr` are fine.
- Use lowercase module names with underscores and `test_<module>.py` beside the
  mirrored source path under `tests/`. Keep tests in arrange, act and assert
  sections separated by blank lines.
- Every Python file in the package, tests, demo, docs and benchmark scripts starts
  with the Apache-2.0 header. Shell scripts carry the one-line copyright notice.
  External-system wrappers open with a docstring naming the system and license.
- Modeling code in `model.py` uses names and docstrings without inline comments.
  Elsewhere, explain contracts and caveats that the code cannot express.
- `inworld` is a namespace package: do not add `src/inworld/__init__.py`.
  Users import `from inworld import voice_isolation` and use its `__all__` API.
  Importing the package must not access the network or import optional demo,
  evaluation or third-party model dependencies.
- Keep one module per concern. Optional dependencies belong behind the extras in
  `pyproject.toml`, with useful errors when they are unavailable.

## Data and reproducibility

- Do not commit machine-specific paths, credentials, private audio, training
  records or experimental checkpoint identifiers. Release names are public, not
  experimental: Voice Isolator v1 (`voice-isolator-v1`,
  `voice-isolator-v1-realtime`) and the v2 the Plans announce. Pass local paths through flags
  or environment variables. Keep scratch work in the gitignored directories.
- Every redistributed recording needs source and license attribution. A
  recording's presence in a working directory does not establish permission to
  publish it.
- Fetch third-party models and dependencies from their upstream releases under
  their own licenses; do not vendor their code or weights.
- The checkpoint checksum in `registry.py` and each golden fixture's
  `meta.json` must agree. Expected golden outputs come from the
  independent training reference implementation; never hand-edit them or
  regenerate them with the implementation being tested.
- Benchmark scripts and wrappers are identified by their exact source bytes.
  Changing them requires the corresponding re-score or re-render. `make format`
  deliberately leaves them untouched; `make format-check` still checks them.
- Generate benchmark figures and reported scores with the benchmark tools;
  do not type in replacement results. The model card is the source of truth for
  model and training disclosures; link to it instead of duplicating them. It is
  also the page Hugging Face shows: its TL;DR and quick start repeat the
  README's, so change them together.

## Validation

Use the commands in [CONTRIBUTING.md](CONTRIBUTING.md). Format and lint touched
Python files and run the relevant tests. A new module needs tests at its mirrored
path. Run the full suite for inference changes and before proposing a release.
Report any skipped optional-dependency or golden-parity checks: the goldens need
the released checkpoints under `models/` (or `VOICE_ISOLATION_GOLDEN_MODEL`).
Tests run on CPU; demo inference can also use a GPU.

[CONTRIBUTING.md](CONTRIBUTING.md#what-a-fresh-checkout-shows) lists what a
fresh checkout shows: the skips to expect, and the platform and Python versions
CI uses. Check a skip, a warning or a failure against it before investigating.
If a check already fails on a clean checkout of `main`, before your change,
report it in one line and carry on with the task you were given; do not
investigate or fix it unasked.

Architecture figures are hand-written SVG with live text in the board charts'
font stack, at the board's width, laid out from the width table in
`docs/build_figures.py`, with an accessible description and light/dark
variants. Preserve that format when changing `docs/build_figures.py`.

## The C# port (`dotnet/`)

`dotnet/` is a self-contained .NET 11 solution (`dotnet/VoiceIsolator.slnx`); it reads the released checkpoints and
the golden fixtures but changes nothing outside `dotnet/`. Build with `dotnet build dotnet/VoiceIsolator.slnx -c Release`
and test with `dotnet test dotnet/tests/VoiceIsolator.Tests -c Release`.

- Float32 results must stay numerically equivalent to the PyTorch network: the layer-by-layer parity tests bound every
  stage's relative error against the dumps of `dotnet/scripts/dump_reference.py`, and the golden fixtures bound the
  output at their own tolerance. Run both after changing numerics; approximations belong behind an option, never in the
  default path.
- The unrolled GEMM micro-kernels (`Kernel512R1..12` in `dotnet/src/VoiceIsolator/Numerics/Gemm.cs`) are generated by
  `dotnet/scripts/gen_kernels.py`; edit the generator and paste its output.
- Packed weights (`*.vimodel`) and NuGet packages are build outputs under the gitignored `dotnet/artifacts/`; never
  commit them.

## Skills

| Skill | Purpose |
|---|---|
| `run-demo` | Launch the demo using its documented example and file-access contract |
| `integrate` | Install the package, load a model, process audio and pin versions |

## Setting up the Voice Isolator for a user

A person asked you to set up the Inworld Voice Isolator: a model that separates
the voice in a recording from everything around it. You install the
`inworld-voice-isolation` Python package, check the machine with
`voice-isolate-doctor`, and clean a first file. The rest of this file is about
changing the repository; you do not need it for this.

Complete every step yourself with your own tools. A step marked
**Ask the person** needs them: ask in one short line, wait until they answer,
then go on. Tell them what to do and what they will get; do not explain how the
model works inside.

What the model does and how well is in the [model card](MODEL_CARD.md). To
build the package into another project or service, the
[integrate skill](.agents/skills/integrate/SKILL.md) covers its API.

### 0. Before you start

- **Already installed?** If `voice-isolate-doctor` runs, skip to [step 3](#3-verify).
- **Use a virtual environment**, never the system Python. Use Python 3.10 or later.
  - With `uv`: `uv venv`, then prefix every `pip` below with `uv`
    (`uv pip install ...`), because a `uv` environment has no `pip` of its own.
  - Without it: `python -m venv .venv`, then activate it.

### 1. Install

Try these in order. When one fails, go to the next.

1. **Inside a clone** (a `pyproject.toml` in this folder names
   `inworld-voice-isolation`). Do not use `make install` here: that is the
   contributor setup.

   ```
   pip install -e .
   ```

2. **From PyPI:**

   ```
   pip install inworld-voice-isolation
   ```

3. **From GitHub**, pinned to the newest tag (`git ls-remote --tags
   https://github.com/inworld-ai/voice-isolator.git` lists them; use `main` if
   there is none):

   ```
   pip install "inworld-voice-isolation @ git+https://github.com/inworld-ai/voice-isolator.git@<tag>"
   ```

Extras, only when they apply:

- `[demo]` (for example `pip install -e ".[demo]"`) when the person wants the
  browser demo with microphone recording.
- `[fast]` only on a machine with an NVIDIA GPU and CUDA. It builds optional
  kernels; skip it if the build fails.

If every route fails, stop and tell the person: "I couldn't install the
Inworld Voice Isolator from PyPI or GitHub. Check the network or proxy
settings and ask me again."

### 2. Weights

Nothing to do. The two models download from Hugging Face
(`inworld/voice-isolator`, about 490 MB together) the first time they are used,
with no account or sign-in, and their checksums are verified.

### 3. Verify

```
voice-isolate-doctor --full
```

It prints one line per check, each starting with `OK`, `INFO`, `WARN` or
`FAIL`, and ends with one summary line. The first run downloads the weights.

- **`All set: …`** means it worked. Go to step 4.
- **Anything else:** each `FAIL` line names one next step. Do the steps you
  can, give the person the rest, and run the check again.

`voice-isolate-doctor` without `--full` uses no network and loads no model; use
it to report the machine's state, and `--json` for a machine-readable copy.

### 4. First result

Use the noisy example. In a clone it is `demo/examples/sample_14.wav`;
otherwise download it:

```
curl -LO https://raw.githubusercontent.com/inworld-ai/voice-isolator/main/demo/examples/sample_14.wav
voice-isolate sample_14.wav sample_14_clean.wav
```

This writes the voice to `sample_14_clean.wav` and the removed background to
`sample_14_clean_background.wav`. Tell the person where both files are.

**Ask the person:** "Want to try your own recording? Give me its path."

For a recording they give you, run the same command with their file and an
output name beside it. Options are in the
[README](README.md#quick-start).

### 5. Finish

Tell the person, in one or two lines, for example:

> The Voice Isolator is ready on the Metal GPU. The clean voice is in
> sample_14_clean.wav and the removed background in
> sample_14_clean_background.wav. Give me a file to clean, or say "open the
> demo" for the microphone version.

Name the device the doctor reported. To open the demo, follow the
[demo README](demo/README.md).

### Rules

- Start the demo locally. Add `--share` (a public link) only when the person
  asked for one.
- Keep the person's audio on this machine: do not upload it anywhere.
- Never ask the person to paste a Hugging Face token or any other secret into
  the chat.

### If something goes wrong

| What you see | What to do |
|---|---|
| Every install route in step 1 fails | Check the network or proxy, then try again. Give the person the message in step 1. |
| `FAIL … inworld/voice-isolator is not reachable` | Hugging Face refused the download. Check that `huggingface.co` opens from this machine, or pass local model files with `--offline-model` and `--online-model`. |
| `FAIL … cannot reach huggingface.co` | No network, a proxy, or `HF_HUB_OFFLINE` is set. Fix the connection, or pass local model files. |
| `FAIL … checksum mismatch` | Delete the file the line names and run `voice-isolate-doctor --full` again. |
| `WARN device: cpu: upgrade torch to 2.6 or later` | On an Apple-silicon Mac, `pip install --upgrade torch torchaudio` to use the Metal GPU. |
| `[fast]` fails to build | Skip it: it is optional and only speeds up NVIDIA GPUs. |
| `FAIL … failed on mps` or `on cuda` | Run with `--device cpu`. Report the error line to the person. |
| The demo cannot use the microphone | Open it at `http://127.0.0.1:7860` (or over HTTPS); browsers allow the microphone only there. |

### Remove the Voice Isolator

- `pip uninstall inworld-voice-isolation` (with `uv`: `uv pip uninstall`).
- Delete the weight files at the paths `voice-isolate-doctor` prints on its
  `cached at` lines.
- Delete the virtual environment you created.
