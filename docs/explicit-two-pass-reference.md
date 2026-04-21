# Explicit Two-Pass Reference

Workflow file:
- [explicit_two_pass_reference.json](/mnt/d/ComfyUI/custom_nodes/ComfyUI-GigaHires/example_workflows/explicit_two_pass_reference.json)

This workflow is intentionally not fancy. It exists to show the structure of hires fix in plain Comfy terms.

## The three parts

### 1. Normal pass

This is just standard txt2img:
- checkpoint
- prompt / negative
- empty latent
- first `KSampler`

That gives you the base latent and base decoded image.

### 2. Upscale

There are two example branches in the workflow.

Latent branch:
- `LatentUpscaleBy`

Image-upscaler branch:
- `VAEDecode`
- `UpscaleModelLoader`
- `ImageUpscaleWithModel`
- `VAEEncode`

### 3. Second pass

This is the part people usually mean by "hires fix", not just the upscale itself.

The second pass is:
- take the upscaled latent
- run a second `KSampler`
- use lower `denoise` than the first pass

So the second pass is basically an img2img-style refinement pass on the bigger latent.

## Why this is useful

If the workflow is split out like this, you can see exactly where each knob belongs:

- first pass controls composition and broad look
- upscale controls size and pre-detail structure
- second pass controls how much new detail gets introduced

## Recommendation

I think your subgraph idea is the better direction.

Instead of one giant hires node, the cleaner UX is probably:
- a few small reusable nodes
- one example workflow / group / subgraph that wires them together
- only the important knobs exposed at the group level

That would keep the process understandable while still letting you break out of the box when needed.
