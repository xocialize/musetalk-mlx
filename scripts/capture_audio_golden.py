"""Capture the PyTorch audio oracle (whisper-tiny path) for Gate C parity.

Mirrors musetalk/utils/audio_processor.py exactly:
  wav -> WhisperFeatureExtractor (mel) -> whisper.encoder(output_hidden_states) stacked
       -> get_whisper_chunk -> per-frame [50, 384] -> (UNet cross-attn states)

Saves the mel (to inject into the MLX encoder), the stacked hidden states, and the
final chunked features. Also dumps the encoder state_dict key inventory.
"""
import math
from pathlib import Path

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor, WhisperModel

ROOT = Path(__file__).resolve().parents[1]
WH_DIR = ROOT / "weights" / "whisper-tiny"
WAV = ROOT / "refs" / "MuseTalk" / "data" / "audio" / "yongen.wav"
OUT = ROOT / "goldens"

fe = AutoFeatureExtractor.from_pretrained(str(WH_DIR))
whisper = WhisperModel.from_pretrained(str(WH_DIR)).eval()

# dump encoder key inventory (raw, from state_dict)
enc_keys = [k for k in whisper.state_dict() if k.startswith("encoder.")]
(OUT / "whisper_encoder_key_inventory.txt").write_text("\n".join(enc_keys))
print(f"encoder tensors: {len(enc_keys)}")

# ---- get_audio_feature ----
wav, sr = librosa.load(str(WAV), sr=16000)
assert sr == 16000
seg_len = 30 * sr
segments = [wav[i:i + seg_len] for i in range(0, len(wav), seg_len)]
mels = [fe(s, return_tensors="pt", sampling_rate=sr).input_features for s in segments]
librosa_length = len(wav)
print(f"segments={len(segments)} mel0={tuple(mels[0].shape)} librosa_length={librosa_length}")

# ---- get_whisper_chunk (audio_processor.py) ----
fps, audio_fps, L, R = 25, 50, 2, 2
feat_len_per_frame = 2 * (L + R + 1)        # 10
with torch.no_grad():
    feats = []
    for m in mels:
        hs = whisper.encoder(m, output_hidden_states=True).hidden_states   # tuple len 5
        feats.append(torch.stack(hs, dim=2))                               # [1, seq, 5, 384]
    whisper_feature = torch.cat(feats, dim=1)
stacked = whisper_feature.cpu().numpy()      # [1, seq, 5, 384]  (pre-chunk encoder output)

idx_mult = audio_fps / fps
num_frames = math.floor((librosa_length / sr) * fps)
actual_length = math.floor((librosa_length / sr) * audio_fps)
wf = whisper_feature[:, :actual_length, ...]
pad = math.ceil(idx_mult)
wf = torch.cat([torch.zeros_like(wf[:, :pad * L]), wf, torch.zeros_like(wf[:, :pad * 3 * R])], 1)
prompts = []
for fi in range(num_frames):
    ai = math.floor(fi * idx_mult)
    clip = wf[:, ai: ai + feat_len_per_frame]
    assert clip.shape[1] == feat_len_per_frame
    prompts.append(clip)
prompts = torch.cat(prompts, dim=0)          # [T, 10, 5, 384]
prompts = rearrange(prompts, "b c h w -> b (c h) w").cpu().numpy()   # [T, 50, 384]

np.savez(
    OUT / "audio_golden.npz",
    mel0=mels[0].cpu().numpy(), stacked=stacked, chunks=prompts,
    num_frames=np.array(num_frames), actual_length=np.array(actual_length),
)
print(f"stacked={stacked.shape} chunks={prompts.shape} num_frames={num_frames}")
print("saved goldens/audio_golden.npz")
