"""Run the official MAGI-1 4.5B-distill (bf16) text-to-video pipeline inside ComfyUI, without the
compiled kernels it normally needs.

The SandAI-org/MAGI-1 repo is cloned next to this package (custom_nodes/MAGI-1) and its `inference`
package is imported lazily at run time, after magi_shims has registered pure-PyTorch stand-ins for
flash_attn / flashinfer / triton. Nothing from that repo's requirements.txt is pinned or built.

Expected weights under <models>/magi1/ (staged by the workflow's concept_mapping):
    t5/t5-v1_1-xxl/...                                  google t5-v1_1-xxl encoder (.bin, fp32)
    vae/{config.json, diffusion_pytorch_model.safetensors}
    4.5B_distill/inference_weight.distill/model-0000N-of-00002.safetensors (+ index json)
"""
import gc
import json
import os
import sys
import tempfile

import torch

import folder_paths

HERE = os.path.dirname(os.path.abspath(__file__))
MAGI_REPO_NAME = "MAGI-1"
FPS = 24  # the model's native rate; only used internally (VAE tile length). The workflow sets playback fps.
CHUNK = 24  # frames per autoregressive chunk

_NODE_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _magi_root():
    for cand in (os.path.join(_NODE_ROOT, MAGI_REPO_NAME), os.path.join(_NODE_ROOT, "magi-1")):
        if os.path.isdir(os.path.join(cand, "inference")):
            return cand
    raise RuntimeError(f"MAGI-1 repo not found next to {HERE} (expected {_NODE_ROOT}/{MAGI_REPO_NAME}/inference)")


def _build_config(weights_dir, num_frames, h, w, steps, seed):
    root = _magi_root()
    with open(os.path.join(root, "example", "4.5B", "4.5B_distill_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    rc = cfg["runtime_config"]
    rc.update(
        seed=seed, num_frames=num_frames, video_size_h=h, video_size_w=w, num_steps=steps, fps=FPS,
        load=os.path.join(weights_dir, "4.5B_distill"),
        t5_pretrained=os.path.join(weights_dir, "t5"),
        vae_pretrained=os.path.join(weights_dir, "vae"),
        t5_device="cuda:0",
    )
    cfg["engine_config"]["fp8_quant"] = False
    path = os.path.join(tempfile.mkdtemp(prefix="magi1_"), "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def _prepare_environment(root):
    # single-process "distributed" setup the pipeline insists on
    env = {
        "MASTER_ADDR": "localhost", "MASTER_PORT": "6009", "GPUS_PER_NODE": "1", "NNODES": "1",
        "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
        "PAD_HQ": "1", "PAD_DURATION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OFFLOAD_T5_CACHE": "true", "OFFLOAD_VAE_CACHE": "true",
        "SPECIAL_TOKEN_PATH": os.path.join(root, "example", "assets", "special_tokens.npz"),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_magi():
    import magi_shims  # sibling module; ComfyUI puts this package dir on sys.path via its loader

    root = _magi_root()
    _prepare_environment(root)
    # transformers probes for flash_attn when it first loads; do that before the fakes exist
    from transformers import AutoTokenizer, T5EncoderModel  # noqa: F401

    magi_shims.install()
    try:
        from inference.common import MagiConfig, set_random_seed
        from inference.infra.distributed import dist_init
        from inference.model.dit import get_dit
        from inference.model.dit import dit_module
        from inference.pipeline import prompt_process
        from inference.pipeline.prompt_process import get_txt_embeddings
        from inference.pipeline.video_generate import generate_per_chunk
        from inference.pipeline.video_process import post_chunk_process
    finally:
        magi_shims.uninstall()

    magi_shims.patch_dit_module(dit_module)

    # T5-XXL in bf16 (upstream loads fp32, ~19 GB of VRAM before the DiT is even loaded)
    from inference.model.t5 import T5Embedder

    def _t5_bf16(model_cache_dir, model_device, model_max_length):
        return T5Embedder(device=model_device, local_cache=True, cache_dir=model_cache_dir,
                          torch_dtype=torch.bfloat16, model_max_length=model_max_length)

    prompt_process._t5 = _t5_bf16
    return MagiConfig, set_random_seed, dist_init, get_dit, get_txt_embeddings, generate_per_chunk, post_chunk_process


class Magi1TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A red fox trots through fresh snow at dawn, cinematic."}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
                "width": ("INT", {"default": 0, "min": 0, "max": 1920, "step": 16}),
                "height": ("INT", {"default": 0, "min": 0, "max": 1920, "step": 16}),
                "num_frames": ("INT", {"default": 48, "min": 24, "max": 480, "step": 24}),
                "num_steps": ("INT", {"default": 12, "min": 8, "max": 64, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "Magi1"

    def generate(self, prompt, seed, width, height, num_frames, num_steps):
        if not torch.cuda.is_available():
            raise RuntimeError("MAGI-1 needs a CUDA GPU")
        w = max(16, (width or 512) // 16 * 16)
        h = max(16, (height or 512) // 16 * 16)
        frames = max(CHUNK, round(num_frames / CHUNK) * CHUNK)
        weights = os.path.join(folder_paths.models_dir, "magi1")

        (MagiConfig, set_random_seed, dist_init, get_dit, get_txt_embeddings,
         generate_per_chunk, post_chunk_process) = _import_magi()

        config = MagiConfig.from_json(_build_config(weights, frames, h, w, num_steps, seed))
        set_random_seed(seed)
        if not torch.distributed.is_initialized():
            dist_init(config)

        with torch.no_grad():
            caption_embs, emb_masks = get_txt_embeddings(prompt, config)
            gc.collect()
            torch.cuda.empty_cache()
            dit = get_dit(config)
            try:
                chunks = [post_chunk_process(c, config)
                          for c in generate_per_chunk(model=dit, prefix_video=None,
                                                      caption_embs=caption_embs, emb_masks=emb_masks)]
            finally:
                del dit
                gc.collect()
                torch.cuda.empty_cache()
        video = torch.cat(chunks, dim=0)  # [T, C, H, W], 0..255
        images = video.permute(0, 2, 3, 1).float().div(255.0).clamp(0, 1).cpu()
        return (images,)


NODE_CLASS_MAPPINGS = {"Magi1TextToVideo": Magi1TextToVideo}
NODE_DISPLAY_NAMES = {"Magi1TextToVideo": "MAGI-1 Text to Video (4.5B distill)"}
NODE_DISPLAY_NAME_MAPPINGS = NODE_DISPLAY_NAMES
