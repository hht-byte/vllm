# AutoGaze adapter for Qwen3.5

This opt-in adapter applies
[AutoGaze](https://github.com/NVlabs/AutoGaze) to Qwen3.5 video inputs without
changing the stock Qwen3.5 implementation or its weights. It follows the
AutoGaze paper and `INTEGRATION.md`, with NVILA-HD-Video as the end-to-end
reference.

The adapter makes the following model-specific translations:

- AutoGaze frame-level selections are unioned across the frames consumed by a
  Qwen temporal tubelet.
- Every gaze is expanded to a complete Qwen
  `spatial_merge_size x spatial_merge_size` patch group. This preserves the
  pretrained patch-merger layout instead of merging unrelated patch rows.
- Qwen learned and rotary vision positions are computed for the full
  multi-scale layout, then gathered for the selected patches.
- Vision attention supports AutoGaze block-causal, causal, and bidirectional
  semantics. The default is block-causal.
- Prompt placeholders and Qwen MRoPE positions use the actual number and
  coordinates of selected output tokens.

## Install AutoGaze

AutoGaze stays an external dependency so the upstream implementation and
checkpoints can be used directly. Install it in the same vLLM environment:

```bash
git clone https://github.com/NVlabs/AutoGaze.git
uv pip install -e ./AutoGaze --no-deps
```

`--no-deps` is intentional: the current AutoGaze package metadata pins an
older Transformers release, while this vLLM branch uses Transformers 5.x.
Install any missing AutoGaze runtime packages through `uv` without replacing
vLLM's Transformers or PyTorch versions.

## Enable the adapter

The plugin is dormant unless explicitly enabled:

```bash
export VLLM_AUTOGAZE_ENABLED=1
export VLLM_AUTOGAZE_MODEL_ID=nvidia/AutoGaze
export VLLM_AUTOGAZE_SCALES=64+128+224+448
export VLLM_AUTOGAZE_GAZING_RATIO=0.1
export VLLM_AUTOGAZE_TASK_LOSS=0.7
vllm serve Qwen/Qwen3.5-9B-Instruct
```

Use `VLLM_PLUGINS=autogaze_qwen3_5` when the installation restricts the set of
vLLM plugins to load.

The available settings are:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_AUTOGAZE_ENABLED` | `0` | Enable the Qwen3.5 adapter. |
| `VLLM_AUTOGAZE_MODEL_ID` | `nvidia/AutoGaze` | AutoGaze checkpoint. |
| `VLLM_AUTOGAZE_SCALES` | `64+128+224+448` | Qwen multi-scale input sizes. |
| `VLLM_AUTOGAZE_GAZING_RATIO` | `0.1` | Maximum gaze ratio. |
| `VLLM_AUTOGAZE_TASK_LOSS` | `0.7` | Reconstruction-loss stop threshold; use `none` to disable. |
| `VLLM_AUTOGAZE_ATTN_TYPE` | `block_causal` | `block_causal`, `causal`, or `bidirectional`. |
| `VLLM_AUTOGAZE_FRAME_INDEPENDENT` | `0` | Restrict vision attention to the same frame. |
| `VLLM_AUTOGAZE_DEVICE` | `cuda` | Device used by the official AutoGaze model. |

Each scale must be divisible by
`vision_config.patch_size * vision_config.spatial_merge_size`. The defaults
target Qwen3.5 configurations with a 16-pixel patch and merge size 2.

## Scope and execution model

The official AutoGaze model runs in the multimodal processor, matching the
published NVILA-HD-Video integration. Qwen patchification keeps the full
multi-scale pixel layout in the processor output, but patch projection and all
vision transformer blocks only process gazed patches.

The adapter currently targets video inputs on Qwen3.5 dense and MoE model
classes. Images and disabled runs retain the original vLLM path. Encoder CUDA
graphs and data-parallel vision-tower sharding are not used for the custom
masked video path. High-resolution spatial tiling can be performed upstream,
as in NVILA-HD-Video; this adapter square-resizes each video item before the
multi-scale Qwen processor.
