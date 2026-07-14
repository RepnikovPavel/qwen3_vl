import os, sys, gc, time
os.environ.setdefault("QWEN3_FP8_KERNEL_DIR", "/tmp/fp8_patched")
os.environ.setdefault("DISABLE_NETWORK_GUARD", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("CKPTDIR", "/models")
sys.path.insert(0, "/opt/qwen3_vl")
import torch
from qwen3_vl_offline import Qwen3VLRuntime
from pathlib import Path
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
def main():
    print("=== GPU before ===")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU{i}: {torch.cuda.memory_allocated(i)/1024**2:.1f} MiB used")
    print("Loading 8B FP8 runtime...")
    t0 = time.time()
    rt = Qwen3VLRuntime(model_size="8b", device="cuda", ckpt_dir="/models", gpu_placement="single")
    print(f"Loaded in {time.time()-t0:.1f}s")
    try:
        from transformers.integrations import finegrained_fp8 as hf_fp8
        hf_fp8._triton_available = None
        print("reset HF FP8 state")
    except: pass
    img = "/state/test_scene.png"
    print(f"Prepare attached image: {img}")
    media = rt.prepare_media([("image", img)], 640)
    print(f"Media prepared, items: {len(media)}")
    prompts = [
        "Describe the scene in one short sentence.",
        "What objects are visible and where are they located?",
        "Focus on spatial layout: lanes, road, vehicles positions.",
    ]
    for p in prompts:
        print(f"\n=== Prompt: {p}")
        t1 = time.time()
        try:
            res, _ = rt.infer(media_inputs=[("image", img)], prompt=p, max_new_tokens=40, do_sample=False)
            txt = getattr(res, "text", str(res)).strip()
            tps = getattr(res, "tokens_per_second", 0)
            print(f"Output: {txt[:200]}")
            print(f"t/s: {tps:.1f} (took {time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"Generate error: {type(e).__name__}: {str(e)[:150]}")
        cleanup()
    print("\n=== GPU after cleanups ===")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU{i}: {torch.cuda.memory_allocated(i)/1024**2:.1f} MiB used")
    print("Test complete.")
if __name__ == "__main__":
    main()
