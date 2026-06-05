"""Phase 5 — realtime perf + fp16 parity for the MuseTalk MLX generation core.

(1) fp16 vs fp32 face-gen parity (production dtype sanity).
(2) batched throughput on GPU at several batch sizes (frames/sec, peak mem).
"""
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline   # noqa: E402

mx.set_default_device(mx.gpu)
g = np.load(ROOT / "goldens" / "pipeline_golden.npz")
chunk = np.load(ROOT / "goldens" / "audio_golden.npz")["chunks"][:1].astype(np.float32)

# fp32 reference
pipe = MuseTalkPipeline.from_pretrained(ROOT / "weights")
recon32 = pipe.generate_faces(mx.array(g["latents"]), mx.array(chunk))

# fp16 parity
pipe.astype(mx.float16)
recon16 = pipe.generate_faces(mx.array(g["latents"]).astype(mx.float16),
                              mx.array(chunk).astype(mx.float16))
d_golden = np.abs(recon16.astype(np.int16) - g["recon"].astype(np.int16))
d_fp32 = np.abs(recon16.astype(np.int16) - recon32.astype(np.int16))
print(f"[fp16 parity] recon16 vs torch-golden: max|Δ|={d_golden.max()} mean={d_golden.mean():.3f}")
print(f"[fp16 parity] recon16 vs mlx-fp32:      max|Δ|={d_fp32.max()} mean={d_fp32.mean():.3f}")

# throughput (fp16, GPU)
N = 64
latents = mx.array(np.repeat(g["latents"], N, 0)).astype(mx.float16)
chunks = mx.array(np.repeat(chunk, N, 0)).astype(mx.float16)
print("\n[throughput fp16, GPU]")
for bs in (1, 4, 8, 16, 32):
    pipe.run_batched(latents[:bs * 2], chunks[:bs * 2], batch_size=bs)  # warmup
    mx.synchronize()
    t = time.time()
    pipe.run_batched(latents, chunks, batch_size=bs)
    mx.synchronize()
    dt = time.time() - t
    fps = N / dt
    print(f"  bs={bs:>2}: {dt*1000/ (N/bs):.1f} ms/batch | {fps:6.1f} frames/s | "
          f"peak {mx.get_peak_memory()/1e9:.2f} GB")
