# Third-Party Notices

## Optional native MPS attention

`corridorkey/native/mps_attention.mm` adapts the unmasked MPSGraph attention
wrapper from [Jason Hurt's attention-mps-torch](https://github.com/jhurt/attention-mps-torch),
revision `d39628f3f292acdf8db8df68807031d56cf6b0ca` (Copyright 2026 Jason Hurt,
Apache License 2.0). The full license is in `corridorkey/native/LICENSE`.
Changes include ARC-safe buffer access, validation of contiguous zero-offset
buffers, compiled-graph feed ordering, an inference-only unmasked interface,
and omission of the optional FP16-intermediate graph rewrite. This component
contains no model weights and does not alter the model-asset licenses.

CorridorKey v2 is licensed under the Corridor Key Licence (see `LICENSE`).
It depends on the third-party components below, which are not owned by
Corridor Digital LLC and remain under their own licenses.

The three foundation models are not redistributed in this repository or in the
CorridorKey Hugging Face repository. `scripts/download_weights.py` fetches them
from their original publishers.

## C-RADIOv4-SO400M

- Publisher: NVIDIA Corporation
- Source: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- License: [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)
- Use: frozen visual feature extractor, downloaded to `vfms/cradio/`

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

## MoGe-2 ViT-B

- Publisher: Microsoft Corporation
- Source: https://github.com/microsoft/MoGe, weights at https://huggingface.co/Ruicheng/moge-2-vitb-normal
- License: MIT (code and weights)
- Use: frozen geometry feature extractor. The model code is vendored in
  `third_party/MoGe/` with its license at `third_party/MoGe/LICENSE`; weights
  are downloaded to `vfms/moge/`.

The vendored MoGe code includes DINOv2 (Meta Platforms, Inc.), licensed under
the Apache License 2.0, also reproduced in `third_party/MoGe/LICENSE`.

## RVM (Recurrent Video MAE), small

- Publisher: Google DeepMind
- Source: https://github.com/google-deepmind/representations4d
- License: Apache License 2.0 (code), Creative Commons Attribution 4.0 (weights)
- Use: frozen recurrent video feature extractor. `corridorkey/rvm.py` is a
  PyTorch port of the reference architecture; weights are downloaded to
  `vfms/rvm/`.
