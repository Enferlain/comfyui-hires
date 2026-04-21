# ReForge Hires Fix Notes

Date: 2026-04-21

Goal:
- build a ComfyUI custom node that is better than a straight port of existing hires-fix / upscale behavior
- keep the good parts of ReForge's txt2img hires pass
- target both quality and wall-clock / VRAM efficiency

## Local sources inspected

ComfyUI:
- `main.py`
- `nodes.py`
- `comfy_extras/nodes_upscale_model.py`
- `custom_nodes/example_node.py.example`
- `custom_nodes/comfyui_ultimatesdupscale/*`

ReForge:
- `/mnt/d/stable-diffusion-webui-reForge/modules/processing.py`
- `/mnt/d/stable-diffusion-webui-reForge/modules/shared.py`
- `/mnt/d/stable-diffusion-webui-reForge/modules/images.py`
- `/mnt/d/stable-diffusion-webui-reForge/modules/sd_samplers_common.py`

## What ReForge hires fix actually does

Core state lives in `StableDiffusionProcessingTxt2Img` in `modules/processing.py`.

Important fields:
- `enable_hr`
- `denoising_strength`
- `hr_scale`
- `hr_upscaler`
- `hr_second_pass_steps`
- `hr_resize_x`
- `hr_resize_y`
- `hr_sampler_name`
- `hr_scheduler`
- `hr_prompt`
- `hr_negative_prompt`
- `hr_checkpoint_name`
- `truncate_x`
- `truncate_y`
- `latent_scale_mode`

### Target size calculation

`calculate_target_resolution()`:
- uses either `hr_scale` or explicit `hr_resize_x` / `hr_resize_y`
- preserves aspect ratio
- when an exact target box does not match the source aspect ratio, it upscales to a covering size and stores latent-space crop amounts in `truncate_x` and `truncate_y`

This means the second pass can overshoot a little and crop back to the requested output size.

### Branch 1: latent upscale

If `hr_upscaler` matches a latent mode from `modules/shared.py`, ReForge uses:
- `torch.nn.functional.interpolate(...)`
- modes: `bilinear`, `bicubic`, `nearest`, `nearest-exact`
- optional `antialias=True` only for bilinear / bicubic modes

Default latent modes:
- `Latent`
- `Latent (antialiased)`
- `Latent (bicubic)`
- `Latent (bicubic antialiased)`
- `Latent (nearest)`
- `Latent (nearest-exact)`

Then ReForge:
- avoids decode/re-encode when it can
- builds image conditioning from the latent directly when inpainting conditioning is not required
- only decodes if it needs img2img-style image conditioning

### Branch 2: pixel upscaler

If `hr_upscaler` is not a latent mode:
- decode low-res latent to RGB
- resize via `images.resize_image(0, image, target_width, target_height, upscaler_name=...)`
- encode the upscaled image back into latent
- run img2img second pass on the new latent

`images.resize_image()` behavior:
- if no learned upscaler is used, it falls back to Pillow LANCZOS
- if scale is greater than `1.0`, it runs the selected external upscaler first
- after that, it still does a final exact-size LANCZOS resize if dimensions do not match exactly

This is good for convenience, but it implies:
- at least one full decode + encode cycle
- potential quality loss from RGB round-trip
- possible mismatch between model-native upscale ratio and final exact-size resample

### Second pass

After either resize branch, ReForge:
- crops latent by `truncate_x` / `truncate_y`
- rebuilds RNG for the new latent shape
- optionally switches sampler, scheduler, checkpoint, and modules
- computes hires-specific prompt conditioning
- runs `sample_img2img(...)` with `hr_second_pass_steps or steps`

Refiner behavior is also aware of hires mode through `p.is_hr_pass` in `modules/sd_samplers_common.py`.

## Relevant ComfyUI primitives

Built-in / core:
- `nodes.py` `LatentUpscale`
- `nodes.py` `LatentUpscaleBy`
- `comfy_extras/nodes_upscale_model.py` `ImageUpscaleWithModel`

Observations:
- ComfyUI already exposes latent interpolation and learned image upscaling
- `ImageUpscaleWithModel` uses tiled model inference and reduces tile size on OOM
- the basic pieces exist, but the orchestration is still manual across many nodes

Existing custom-node reference:
- `custom_nodes/comfyui_ultimatesdupscale`

That package is valuable mostly as a reference for:
- tiled redraw
- seam handling
- mask-blurred refinement windows

It is not the same problem as ReForge hires fix, but it overlaps with "quality left on the table" thinking.

## Quality / performance gaps worth exploring

### 1. Hybrid upscale instead of binary latent-vs-pixel

ReForge chooses one path globally:
- latent interpolation
- or RGB upscaler + VAE encode

Potential upgrade:
- do a cheap latent upscale for global structure
- then optionally inject learned pixel/detail guidance only on selected regions or frequencies
- avoid forcing the entire image through a full RGB round-trip

### 2. Exact-size planning without late truncate crop

ReForge's `truncate_x` / `truncate_y` is practical, but it means:
- overshoot
- crop
- potential detail loss near framing boundaries

Potential upgrade:
- compute a latent target plan that lands exactly on the intended framing
- or expose explicit framing policy: `cover`, `contain`, `preserve composition`, `strict exact`

### 3. Better conditioning after upscale

Current ReForge logic uses a single global second img2img pass.

Potential upgrade:
- adaptive denoise based on edge/detail density
- lower denoise for stable flat areas
- higher denoise for texture-rich or upscaler-hallucinated areas
- optional mask generation from gradient / saliency / high-frequency energy

### 4. Better decode / encode discipline

Expensive parts:
- VAE decode before pixel upscale
- VAE encode after pixel upscale

Potential upgrade:
- tiled VAE encode/decode path
- optional approximate VAE for planning previews, full VAE for final
- cache intermediate RGB / latent pairs when only pass-2 settings change

### 5. Multi-stage upscale schedules

Current ReForge hires fix is mostly:
- generate low-res
- upscale once
- denoise once

Potential upgrade:
- schedule like `1.0 -> 1.5 -> 2.0`
- decreasing denoise across stages
- optional sampler/scheduler changes per stage
- optional learned upscaler only on the last stage

### 6. Seam-aware detail redraw without full USDU complexity

Potential upgrade:
- identify only problematic windows after upscale
- run localized redraw where sharpness / coherence falls below threshold
- use overlap-aware feather masks
- skip seam-fix work when the upscaled latent already looks clean

### 7. Pass-specific model strategy

ReForge already supports different checkpoint / modules / sampler on hires pass.

Potential upgrade in Comfy:
- first-class pass-1 / pass-2 parameter blocks
- explicit "same as pass 1" vs overridden values
- easier experimentation than current manual graph duplication

## Likely first implementation target

A practical first node should probably be a coordinator node, not a new sampler:

Inputs:
- `model`
- `clip`
- `vae`
- `positive`
- `negative`
- `latent` or `image`
- upscale strategy
- target sizing policy
- pass-2 denoise / steps / sampler settings
- optional upscale model

Outputs:
- final latent
- final image
- optional debug image / debug metadata

Suggested v1 modes:
- `latent_only`
- `pixel_only`
- `hybrid_latent_then_model`

Suggested debug outputs:
- planned target size
- actual latent size before and after crop
- whether decode / encode happened
- elapsed timings per stage

## First implementation opinion

Best starting point for v1:
- keep the internal flow simple
- support a high-quality latent upscale path first
- support a second path that uses Comfy's `ImageUpscaleWithModel`-style tiled model upscale
- make the node return debug metadata so we can benchmark design choices quickly

Avoid for v1:
- full Ultimate SD Upscale style redraw engine
- too many magic heuristics hidden from the user
- silent quality tradeoffs

## Immediate next steps

1. Build a minimal coordinator node that reproduces ReForge's two-pass logic inside ComfyUI.
2. Add instrumentation for timing, latent shape, decode/encode count, and chosen branch.
3. Add one quality improvement beyond ReForge on day one.
4. Benchmark against:
   - plain latent upscale + KSampler
   - upscale-model -> VAE encode -> KSampler
   - ReForge-like reproduction path

## Day-one improvement candidates

Strong candidates:
- hybrid latent pre-upscale followed by optional learned model refinement
- exact framing policy that avoids ReForge-style late crop when possible
- tiled VAE path for big outputs
- adaptive denoise mask for second pass

Most realistic first upgrade:
- exact framing + hybrid path + debug instrumentation
