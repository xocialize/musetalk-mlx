# MuseTalk 1.5 → MLX — Pre-flight & Port Plan

> Single source of truth for the `musetalk-mlx` port. Mirrors the `zonos-mlx` convention
> (CONFIRM gates → scaffold → PyTorch oracle → per-component parity gates → e2e → realtime →
> quantize → publish). Per-port status rolls up into `XDocs/PORTS-STATUS.md`.

**Started:** 2026-06-04 · **Owner:** Dustin / MVS Collective (xocialize-code)
**Hardware:** MacBook Pro M5 Max 128 GB
**v1 scope (chosen):** Core neural port **+ realtime pipeline** (Python). MLX-Swift mirror deferred post-WWDC.

---

## Upstream (pinned)

- **Repo:** https://github.com/TMElyralab/MuseTalk
- **Commit:** `0a89dec45a0192b824e3cf4daf96c239440c5ed8` (cloned to `refs/MuseTalk`, code only — weights from HF)
- **Weights:** `TMElyralab/MuseTalk` (HF, 6.8 GB) — `musetalkV15/{unet.pth, musetalk.json}`
- **Component weights:** `stabilityai/sd-vae-ft-mse` (VAE), HF `whisper-tiny` (audio), DWPose, S3FD, face-parse-bisent (CPU preprocessing — run as-is)

## CONFIRM gates (all PASS — 2026-06-04)

1. **License** ✅ MIT, commercial-OK. Upstream `LICENSE`: *"no limitation for both academic and
   commercial usage"*; trained models *"available for any purpose, even commercially."* Dependency
   models keep their own (all permissive). Test **data** is non-commercial-research-only (irrelevant to a port).
2. **Port-status** ✅ No existing MLX port (not on mlx-community / mlx-audio / mlx-vlm; no `-mlx` fork). Net-new.
3. **Config truth** ✅ Pulled real `musetalkV15/musetalk.json` (see below). Pinned commit.
4. **Tier** → **Tier 3** (multi-component) but light: single forward pass, NOT diffusion (no denoise loop, no CFG schedule).

---

## Architecture (the oracle)

MuseTalk is **single-step latent inpainting**, *not* a diffusion model. One UNet forward per frame
regenerates the masked (lower-face/mouth) region conditioned on audio.

```
audio.wav ─► HF WhisperModel(tiny) encoder ─► 384-d features ─► chunk[2,2] ─► PositionalEncoding ─┐
                                                                                                   ▼ encoder_hidden_states (cross-attn, dim=384)
face frame ─► detect+crop 256² ─► VAE.encode(half-masked) ──┐                                       │
            └────────────────► VAE.encode(full reference) ──┴─ cat dim=1 ─► 8-ch latent ─► UNet2DConditionModel(t=0) ─► 4-ch pred latent
                                                                                                   │
                                                                          VAE.decode ─► 256² face ─► blend/paste back into frame
```

### UNet — `musetalkV15/musetalk.json` (verbatim)
```jsonc
{ "_class_name": "UNet2DConditionModel", "_diffusers_version": "0.6.0.dev0",
  "act_fn": "silu", "attention_head_dim": 8,            // ⚠ MISNOMER: heads-style, real head_dim = ch//8
  "block_out_channels": [320, 640, 1280, 1280],
  "center_input_sample": false, "cross_attention_dim": 384,   // ← whisper-tiny feat dim, NOT 768 text
  "down_block_types": ["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
  "downsample_padding": 1, "flip_sin_to_cos": true, "freq_shift": 0,
  "in_channels": 8,                                      // 4 masked-target latent ⊕ 4 reference latent
  "layers_per_block": 2, "mid_block_scale_factor": 1, "norm_eps": 1e-05, "norm_num_groups": 32,
  "out_channels": 4, "sample_size": 64,                 // nominal; 256px face ÷ 8 = 32² actual latent
  "up_block_types": ["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"] }
```
Forward (from `scripts/inference.py:203`): `unet.model(latent_batch, timesteps=[0], encoder_hidden_states=pe(whisper_batch)).sample`.
Fixed timestep 0 — but the time-embedding path still runs (embeds t=0). `PositionalEncoding(d_model=384)`
(sinusoidal, `musetalk/models/unet.py:12`) is added to audio features *before* cross-attn.

### VAE — stock `AutoencoderKL` (`stabilityai/sd-vae-ft-mse`)
4-channel latent, `scaling_factor=0.18215`. `get_latents_for_unet` (`musetalk/models/vae.py:110`):
encode(half-masked ref) → masked_latents [1,4,32,32]; encode(full ref) → ref_latents; `cat(dim=1)` → [1,8,32,32].
Mask = upper half kept, lower half (mouth) zeroed. Normalize mean/std 0.5. **NOT in DiffusionKit** (SD3/FLUX only) → manual Tier-2 port.

### Audio — HF `WhisperModel` (tiny) encoder
v1.5 uses `transformers.WhisperModel.from_pretrained` + feature extractor (`scripts/inference.py:64-68`),
**not** the bundled whisper. 384-d hidden states → `get_whisper_chunk` window `[2,2]` (2 left + center + 2 right)
→ reshape `(-1, 384)`. **Reuse existing `whisper-mlx` / `tts-validation` mlx-whisper encoder.**

---

## Port phases & gates

| # | Phase | Deliverable | Gate |
|---|-------|-------------|------|
| 0 | **Pre-flight** | scaffold + `refs/` + `.venv` + PyTorch oracle + goldens capture script | this doc; gates 1–4 PASS |
| 1 | **VAE** `models/vae.py` ✅ | AutoencoderKL encode+decode in MLX (NHWC, stock-MLX conv) | **Gate A PASS**: encode mean **1.69e-5** / logvar 4.9e-6 · decode **3.43e-5** (cpu fp32); GPU roundtrip clean |
| 2 | **UNet** `models/unet.py` ✅ | SD1.x `UNet2DConditionModel` (8→4ch, cross_attn=384, t=0) | **Gate B PASS**: full forward **1.41e-6** (cpu fp32); GPU clean, ~190ms/frame fp32 bs=1 |
| 3 | **Audio** `whisper/` ✅ | whisper-tiny encoder + `get_whisper_chunk` + sinusoidal PE | **Gate C PASS**: encoder **1.6e-5**, chunk **0.0** (exact), PE 3.1e-6 (cpu fp32) |
| 4 | **Pipeline** `pipeline_mlx.py` ✅ | mask → encode×2 → cat8 → UNet(t=0) → decode → blend/paste | **Gate D-core PASS**: face-level recon **≤2/255** vs torch. **Gate D-visual PASS**: real 268-frame clip (faithful DWPose-onnx + S3FD crops, bisenet blend, audio mux) → clean lip-synced video; MLX-vs-torch on real crops **max|Δ|=3 / mean 0.023** ⇒ SyncNet ≡ upstream by construction |
| 5 | **Realtime** ✅ | batched fp16 frames, `mx.eval` boundaries | **Gate E PASS**: bs=8 → **34 fps** (>25fps realtime), bs=4 → 30.6 fps; peak 7 GB; fp16 recon mean\|Δ\|=0.43 vs torch |
| 6 | **Quantize + publish** ✅ | int8/int4 UNet Linears (VAE+audio fp16); `dist/` + card | **Gate F PASS**: UNet cosine **q8 1.00000 / q4 0.99985**; recon mean\|Δ\| fp16 0.32 / q8 0.41 / q4 2.74. **Published `mlx-community/MuseTalk-1.5-{fp16,q8,q4}`** (torch-free, self-contained) |

### Parity discipline (carried from zonos/lens lessons)
- **Run correctness parity on `mx.cpu`** — Apple-GPU fp32 matmul is tf32-like (~3.8e-3/matmul); CPU is bitwise-ish.
- **Gate on RELATIVE error**, not absolute max_abs (SD VAE / UNet have large-magnitude activations).
- Generate random inputs on numpy, inject into BOTH sides (MLX RNG ≠ torch RNG).
- PyTorch is optional-dep (`pip install -e ".[parity]"`); end users never pull torch.

### Known traps to honor (from skill + this config)
- `attention_head_dim: 8` is **num-heads-style** → real per-head dim = `block_ch // 8` per level.
- diffusers `0.6.0.dev0` UNet — port THAT version's block decomposition (resnet/attn order), not a newer one.
- Conv weight layout PT `(O,I,H,W)` → MLX `(O,H,W,I)`; materialize (`mx.eval`) every tensor before save.
- Single-step but time-embedding path still executes at t=0 — don't drop it.

---

## Lessons (append as we learn — fold into `mlx-porting` skill at the end)

- **M1 (Phase 1, VAE)** — **The raw `sd-vae-ft-mse` safetensors uses the *pre-rename* diffusers
  attention keys** `query/key/value/proj_attn`, NOT `to_q/to_k/to_v/to_out.0`. `from_pretrained`
  remaps them on load, so a state_dict dumped *after* loading (the golden-key inventory) shows the
  NEW names — but a loader reading the file directly sees the OLD names. Fix: rename map in
  `utils/weights.py`. **General rule: dump the key inventory from the raw safetensors file, not from
  `model.state_dict()` post-load.** (`group_norm` was already current; only q/k/v/proj renamed.)
- **M2 (Phase 1, VAE)** — Keep the **public encode/decode API fully NCHW** (diffusers-isomorphic) even
  though MLX runs NHWC internally. Exposing NHWC latents from `encode` while `decode` expects NCHW
  caused a silent layout mismatch. `DiagonalGaussian` stores NCHW moments; transpose at the boundary only.
- **M3 (Phase 1, VAE)** — diffusers VAE GroupNorm eps is **1e-6**, not the UNet's 1e-5. Use
  `nn.GroupNorm(..., pytorch_compatible=True)` so channel-group ordering matches torch.
- **M4 (Phase 2, UNet)** — **`attention_head_dim: 8` = NUM HEADS, not per-head dim.** Resolved
  empirically: introspected `mod.heads` on every diffusers `Attention` → all 8, constant across
  resolutions (head_dim = ch//8 = 40/80/160). The misnomer trap from the skill, confirmed live.
  Always introspect the loaded model rather than guessing from the config field name.
- **M5 (Phase 2, UNet)** — eps differs *within* the same UNet: resnets + `conv_norm_out` use
  `norm_eps`=**1e-5**; `Transformer2DModel.norm` (the pre-proj_in GroupNorm) uses **1e-6** hardcoded.
  Two different GroupNorm eps in one model — don't unify them.
- **M6 (Phase 2, UNet)** — diffusers `proj_in`/`proj_out` are **1×1 convs** (use_linear_projection=False
  for SD1.x), not Linear; attn `to_q/k/v` carry **no bias**, only `to_out.0` does. Verify bias presence
  from the key set, not the module name. UNet `Downsample2D` uses symmetric `padding=1` (NOT the VAE
  encoder's asymmetric pad-then-stride). Up-block skips: pop `len(resnets)` residuals, **reverse**, concat
  on the channel axis before each up resnet. Result: full-forward parity **1.41e-6** first try.
- **M7 (Phase 3, audio)** — MuseTalk's audio cond = whisper-tiny encoder run with
  `output_hidden_states=True`, **all 5 hidden states stacked** on a new axis: states =
  `(in_layer0, in_layer1, in_layer2, in_layer3, layer_norm(out_layer3))` — i.e. HF collects the input
  to each layer (pre-layer), then the final = `layer_norm(last)`. Per video frame, a window of 10
  encoder steps × 5 hidden = **50 tokens × 384** → that's the cross-attn seq len (matches the UNet
  golden). `k_proj` has no bias (whisper). Encoder parity **1.6e-5**; chunk slicing bit-exact.
- **M8 (Phase 3, audio)** — the `whisper-tiny` HF safetensors is a *full*
  `WhisperForConditionalGeneration` → encoder keys are prefixed **`model.encoder.`**, not `encoder.`.
  Strip the right prefix when loading just the encoder. (Mel extraction still uses HF
  `WhisperFeatureExtractor` — deterministic CPU preproc; torch-free mel is a later follow-up.)
- **M9 (Phase 4, pipeline)** — MuseTalk's masked latent is the encode of a **half-zeroed image**
  (sharp -1 discontinuity at row 128). That synthetic edge produces a single localized error
  *spike* in the latent (max-rel ~1.2e-3) while the bulk mean|Δ|~1e-4 and the unmasked ref-latent
  parities at ~1e-5. Don't gate masked-latent on max-rel — gate on **mean|Δ|** + the strict decoded
  recon (≤2/255). Real e2e correctness shows in the recon, not the latent spike. Also: production
  `get_latents_for_unet` uses `latent_dist.sample()` (stochastic); the MLX port defaults to the
  posterior **mean** (deterministic + cleaner) — note the intentional divergence.
- **M10 (Phase 5, realtime)** — **fp16 GPU matmul output is batch-size-dependent**: the same frame
  decoded in a batch of 4 vs 2 differs by up to ~20/255 on a single pixel (different Metal kernels /
  accumulation order per batch size), while mean\|Δ\| stays ~0.4. Gate fp16 batched paths on **mean**
  abs error, never a single-pixel max. fp32 is stable; only fp16 shows this.
- **M11 (follow-up, torch-free mel)** — replaced HF `WhisperFeatureExtractor` with a self-contained
  MLX log-mel (`whisper/log_mel.py`): STFT n_fft=400/hop=160, **periodic** Hann (`np.hanning(n+1)[:-1]`),
  center+**reflect** pad n_fft//2, power, **drop the last STFT frame** (3001→3000), mel filterbank,
  `log10 → max(·, max-8) → (·+4)/4`. Two parity-critical details: ship HF's *exact* mel filterbank
  (extract `fe.mel_filters`, slaney norm/scale → `assets/mel_filters_80.npy`) rather than recomputing,
  and drop the trailing frame. Result: mel max\|Δ\|=1.3e-5 vs HF; full wav→chunks max\|Δ\|=3.9e-3
  (mean 1.5e-6). **Runtime is now fully torch/transformers-free** (verified with both blocked).
- **M12 (Gate D-visual, DWPose without mmcv)** — **`mmcv`/`mmpose` will not build on macOS+py3.12**
  (old `pkg_resources` calls `pkgutil.ImpImporter`, removed in 3.12; openmmlab ships no macOS wheels,
  China server times out). The faithful escape: **`rtmlib` + `onnxruntime`** runs the *exact* upstream
  DWPose weights (`dw-ll_ucoco_384.onnx` + `yolox_l.onnx` from `yzd-v/DWPose`) — zero compilation, same
  keypoints (`[23:91]` = 68 face landmarks). Pair with the bundled S3FD detector (pure torch, no mmcv)
  for the bbox. Whole faithful crop path runs in the main env. Also: the bisenet face-parse checkpoint
  is legacy-tar format → shim `torch.load(..., weights_only=False)` (torch-2.6+ default flip).
- **Validation-equivalence note** — when the MLX output is pixel-equivalent to the torch reference on
  real data (here ≤3/255), any downstream quality metric (SyncNet) is equal **by construction** — no
  need to run the metric (which here would need the unbundled LatentSync SyncNet arch). The MLX-vs-torch
  pixel diff is a *stronger* port-correctness proof than an absolute SyncNet score.
