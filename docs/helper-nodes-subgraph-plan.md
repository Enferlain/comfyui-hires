# Helper Nodes Reference

Workflow file:
- [helper_nodes_reference.json](/mnt/d/comfyui/custom_nodes/ComfyUI-GigaHires/example_workflows/helper_nodes_reference.json)

This is the workflow version of the direction you described:
- visible normal pass
- visible upscale section
- visible second-pass refine section
- helper nodes only where stock Comfy feels awkward

## What changed from the plain reference

The earlier stock-node reference showed the structure, but it still inherited some stock-node friction.

The helper-node version fixes the main pain points:
- `GigaHires Latent Upscale` gives a direct `scale_by` or exact target size for the latent path
- `GigaHires Image Upscale` gives a direct `scale_by` or exact target size for the image-upscaler path
- `GigaHires Refine Pass` makes the second pass explicit instead of hiding it inside one large hires node

## The three sections

### 1. Normal pass

This is still standard Comfy:
- checkpoint
- prompt / negative
- empty latent
- first `KSampler`

That section is responsible for composition, pose, framing, and the overall look.

### 2A. Latent upscale branch

This is the cleanest hires-fix path when you want to stay in latent space:
- first-pass latent
- `GigaHires Latent Upscale`
- `GigaHires Refine Pass`

Important knobs:
- `latent_mode`
- `scale_by` or `target_width` / `target_height`
- second-pass `steps`
- second-pass `denoise`

### 2B. Image upscale branch

This is the pixel-space path for learned upscalers:
- `VAEDecode`
- `UpscaleModelLoader`
- `GigaHires Image Upscale`
- `VAEEncode`
- `GigaHires Refine Pass`

Important knobs:
- `method`
- `scale_by` or `target_width` / `target_height`
- the loaded `UPSCALE_MODEL`
- second-pass `denoise`

This is the branch that replaces the awkward stock setup where the upscaler path is harder to size the way you want.

## What the second pass actually is

The second pass is just a controlled refinement sample on the bigger latent.

That means:
- the upscale step makes the canvas bigger
- the refine step decides how much fresh detail gets invented

So if the result is too soft, too crunchy, or drifting too much, the first knob to adjust is usually the refine-pass `denoise`, not the upscale itself.

## Subgraph direction

I think this is the right shape for a Comfy subgraph:
- keep the helper nodes small
- keep the workflow readable
- expose only a few face controls on the subgraph

The face controls I would expose first:
- branch choice: latent or image-upscaler path
- `scale_by`
- second-pass `steps`
- second-pass `denoise`
- second-pass sampler / scheduler
- optional upscaler model selector

Everything else can stay inside unless we learn it needs to be promoted.

## Practical next move

Load the helper workflow, test both branches, and tell me which knobs still feel noisy or missing.

Then I can either:
- add one more tiny helper node if a repeated pain point appears, or
- reshape this into a subgraph-ready layout with cleaner group boundaries.
