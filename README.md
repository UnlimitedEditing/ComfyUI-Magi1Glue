# ComfyUI-Magi1Glue

Runs SandAI's MAGI-1 4.5B-distill (bf16) text-to-video inside ComfyUI with no compiled dependencies.

The official `SandAI-org/MAGI-1` repo hard-imports `flash_attn==2.4.2`, `flashinfer` and `triton`. `magi_shims.py`
registers pure-PyTorch equivalents before that repo's `inference` package is imported. Clone MAGI-1 next to this
package (`custom_nodes/MAGI-1`); do not install its requirements.txt. Weights go under `models/magi1/`
(see the docstring in `__init__.py`). One node: `Magi1TextToVideo`. Only bf16 weights (fp8 needs flashinfer).

`python test_shims.py` checks the shims against reference math on CPU.
