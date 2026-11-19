# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dumps the PyTorch network's activations, layer by layer, for the C# port.

The C# parity tests (dotnet/tests) compare every stage of their network with
these arrays: the STFT, the band encoders, each block (with its time half, its
attention projection or forward Mamba layer, and a Mamba block's band half),
the decoder heads, the correction and the output audio. Streaming outputs of the package's own ``Stream`` are dumped for
a few push patterns as well.

    python dotnet/scripts/dump_reference.py \
        --model models/denoiser.safetensors \
        --input tests/fixtures/offline_3s/input.wav \
        --out staging/dotnet-ref/offline
    python dotnet/scripts/dump_reference.py \
        --model models/denoiser_realtime.safetensors --lookahead 2 \
        --input tests/fixtures/offline_3s/input.wav \
        --out staging/dotnet-ref/realtime

Everything runs in float32 on the CPU. Each directory gets ``meta.json``
describing the arrays; the C# tests read ``VOICE_ISOLATOR_REFERENCE_DIR``.
"""

import argparse
import json
import os

import numpy as np
import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import isolator
from inworld.voice_isolation import model as modeling


def _save(out: str, name: str, value: torch.Tensor, names: list) -> None:
    array = value.detach().float().cpu().numpy()
    np.save(os.path.join(out, f"{name}.npy"), np.ascontiguousarray(array))
    names.append({"name": name, "shape": list(array.shape)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--seconds",
                        type=float,
                        default=0.0,
                        help="Keep only the first seconds of the input.")
    parser.add_argument("--streams-only",
                        action="store_true",
                        help="Dump only the streaming outputs (long inputs).")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out, exist_ok=True)

    network = isolator.load_model(args.model, "cpu")
    if args.lookahead is not None:
        network.set_lookahead(args.lookahead)
    wav, rate = audio_lib.load_audio(args.input)
    wav = audio_lib.resample(wav, rate, network.config.sample_rate)
    if args.seconds:
        wav = wav[:int(args.seconds * network.config.sample_rate)]
    names: list = []
    _save(args.out, "input", wav, names)

    if not args.streams_only:
        _dump_layers(network, wav, args.out, names)
    streams = _dump_streams(network, wav, args.out, names)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": os.path.basename(args.model),
                "config": network.config.to_dict(),
                "lookahead_frames": network.lookahead_frames,
                "offline": network.offline,
                "arrays": names,
                "streams": streams
            },
            f,
            indent=1)


def _dump_layers(network, wav: torch.Tensor, out: str, names: list) -> None:
    """Every stage of one forward pass over the whole clip."""
    captured = {}

    def hook(name):

        def capture(module, inputs, output):
            del module, inputs
            captured[name] = output[0] if isinstance(output, tuple) else output

        return capture

    for index, block in enumerate(network.blocks):
        block.register_forward_hook(hook(f"block{index}"))
        if isinstance(block, modeling.DualPathAttention):
            block.time_transformer.register_forward_hook(
                hook(f"block{index}.time"))
            block.time_transformer.attn.register_forward_hook(
                hook(f"block{index}.time.attn"))
        else:
            block.time_mamba.register_forward_hook(hook(f"block{index}.time"))
            block.time_mamba.forward_blocks.register_forward_hook(
                hook(f"block{index}.time.forward"))
            block.freq_mamba.register_forward_hook(hook(f"block{index}.freq"))
    network.denoise_decoder.register_forward_hook(hook("decoder"))

    batch = wav[None]
    with torch.inference_mode():
        spectrum = network.stft(batch)
        encoded = network.encode(spectrum)
        output = network(batch)
        raw = captured["decoder"]
        if not network.offline and network.lookahead_frames:
            raw = modeling.read_ahead(raw, network.lookahead_frames)
        correction = network.correction(raw, spectrum)
    bands, frames = encoded.shape[1], encoded.shape[2]
    _save(out, "stft", torch.view_as_real(spectrum[0]), names)
    _save(out, "encode", encoded[0], names)
    for key, value in captured.items():
        if key == "decoder":
            _save(out, key, value[0], names)
            continue
        # The time halves run on [bands, frames, width] rows; the band halves
        # on [frames, bands, width]. Both are stored as [bands, frames, width].
        if key.endswith(".freq"):
            value = value.reshape(frames, bands, -1).transpose(0, 1)
        else:
            value = value.reshape(bands, frames, -1)
        _save(out, key, value, names)
    _save(out, "correction", torch.view_as_real(correction[0]), names)
    _save(out, "output", output[0], names)


def _dump_streams(network, wav: torch.Tensor, out: str, names: list) -> list:
    """The package's own stream over the clip, for a few push patterns."""
    streams = []
    if network.offline:
        return streams
    live = isolator.VoiceIsolator(network,
                                  device="cpu",
                                  precision="fp32",
                                  mode="streaming",
                                  lookahead_frames=network.lookahead_frames)
    for name, piece, window in (("stream_plain_960", 960,
                                 0), ("stream_windowed_960", 960, None),
                                ("stream_windowed_7001", 7001, None)):
        session = live.stream(window_frames=window)
        with torch.inference_mode():
            pieces = [
                session.push(wav[i:i + piece])
                for i in range(0, len(wav), piece)
            ]
            pieces.append(session.flush())
        _save(out, name, torch.cat(pieces), names)
        streams.append({"name": name, "piece": piece, "window_frames": window})
    return streams


if __name__ == "__main__":
    main()
