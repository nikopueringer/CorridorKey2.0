from pathlib import Path

import torch

from ofx.worker import frame_config


def test_single_frame_mps_worker_keeps_spatial_memory_bounded(tmp_path):
    mps = frame_config(torch.device("mps"), tmp_path, True, 0.25)
    cuda = frame_config(torch.device("cuda"), tmp_path, True, 0.25)
    assert (mps.hann_chunk, mps.hann_stride) == (1, 1)
    assert mps.low_vram and not mps.auto_vram
    assert mps.low_vram_foundation_chunk == mps.low_vram_swin_window_batch == 1
    assert mps.low_vram_mlp_chunk_tokens == 4096
    assert mps.low_vram_full_attn_query_chunk_tokens == 128
    assert cuda.low_vram_foundation_chunk == 4
    assert cuda.low_vram_swin_window_batch == 16
    assert cuda.low_vram_full_attn_query_chunk_tokens == 2048
    assert mps.frame_dir_linear and mps.despill_strength == 0.25
    assert mps.outputs == {"cutout"} and mps.exr_dtype == "float"
