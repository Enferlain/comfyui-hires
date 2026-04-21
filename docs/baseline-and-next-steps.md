# Baseline And Next Steps

Date: 2026-04-21

This note captures what we learned from the first real ComfyUI tests, so we do not have to rediscover it next time.

## Current baseline

The current helper-node workflow is a valid baseline:
- the interactions behave as expected
- the helper-node split is usable
- performance is not obviously worse than ReForge for comparable hires-fix settings

That matters, because it means the project can move on from "is the structure broken?" to "what actual improvements beat the baseline?"

## What the timing tests showed

The major cost is the hires second pass sampler, not the VAE decode.

Measured examples from `GigaHires Refine Pass`:

### 2304 x 1536 image-space target

Latent shape:
- `1 x 4 x 288 x 192`

Settings:
- `12` steps
- `cfg 5.2`
- `res_momentumized_v5`
- `beta`
- `denoise 0.35`

Timing:
- sample: about `27.7s`
- decode: about `1.35s`
- total inside refine pass: about `29.1s`

### 3072 x 2048 image-space target

Latent shape:
- `1 x 4 x 384 x 256`

Settings:
- `12` steps
- `cfg 5.2`
- `res_momentumized_v5`
- `beta`
- `denoise 0.45`

Timing:
- sample: about `55.0s`
- decode: about `2.75s`
- total inside refine pass: about `57.8s`

### Earlier 28-step runs

The 28-step tests confirmed the same pattern:
- sampler time dominates hard
- decode is comparatively small
- bigger latent sizes scale the cost roughly the way we would expect

## Important conclusions

### 1. The current architecture is not the main problem

The helper-node design is not introducing some mysterious giant overhead.

At matched settings, the time is mostly the normal cost of doing a second sample on a much larger latent.

### 2. The second pass budget is the real cost center

The main bottleneck is:
- pass-2 latent size
- pass-2 steps
- pass-2 sampler choice
- pass-2 denoise

So future optimization work should focus there first.

### 3. Latent upscale can stay blocky even with smoother interpolation

This was an important finding:
- `nearest-exact` being blocky at lower denoise was expected
- but `bicubic antialiased` also still looked blocky in testing

That suggests the problem is not just "pick a better interpolation mode."

Instead, it suggests:
- some images simply do not clean up enough with a latent-only resize plus moderate denoise
- the second pass may not be rewriting enough of the resized latent structure
- the image-upscaler path or a future hybrid path may be the better answer for those cases

### 4. More steps did not magically solve the latent blockiness

The tests suggest that simply throwing more hires-pass steps at the latent branch is not a reliable fix.

That means the next quality gains are more likely to come from:
- better branch choice
- better sampler choice
- better denoise strategy
- hybrid approaches

not from "just add more steps."

## Practical working defaults for now

These are not permanent truths, just sane starting points based on current testing.

### Latent branch

Use when:
- upscale amount is moderate
- preserving composition is the priority
- speed still matters

Good starting range:
- `Latent (antialiased)`
- `10-12` steps
- `0.30-0.40` denoise

Notes:
- low-denoise latent upscale can still show chunkiness or blockiness
- very large targets make the latent path expensive fast

### Image-upscaler branch

Use when:
- latent path still looks blocky
- target size is large
- you want cleaner structure before the second pass

Good starting range:
- learned upscaler branch
- `10-14` steps on pass 2
- `0.25-0.35` denoise

Notes:
- this is probably the better branch to prefer once the target size gets very large
- always prefer an external `UpscaleModelLoader` over relying on the internal dropdown if we want clearer control and less ambiguity

## Known things to remember next time

### 1. Debug info is useful now

We added:
- `GigaHires Debug Print`

Use it to inspect:
- refine-pass sample time
- refine-pass decode time
- image-upscaler timing
- whether the upscale model was loaded internally

### 2. The interpolation dropdown is not the whole story

Changing latent interpolation mode matters, but it is not the master key.

If a latent-upscaled image still looks blocky after trying smoother modes, the next move should probably be:
- image-upscaler branch
- or a hybrid experiment

not endless interpolation tweaking.

### 3. Sampler choice is still underexplored

Most current timings used:
- `res_momentumized_v5`
- `beta`

That may be a poor quality-per-time baseline for hires cleanup.

One of the first real comparisons next time should be:
- current pass-2 sampler
- versus `dpmpp_2m + karras`

at matched target size and denoise.

## Best next experiments

These are the highest-value next steps.

### 1. Branch recommendation heuristics

Goal:
- determine when latent path is a good default
- determine when image-upscaler path should be preferred automatically

Likely factors:
- target resolution
- upscale factor
- denoise
- maybe latent area threshold

### 2. Better pass-2 defaults

Goal:
- reach ReForge-level or better quality with fewer wasted steps

Candidates:
- lower default hires-pass steps
- different default sampler/scheduler
- better denoise suggestions by branch

### 3. Hybrid path

This is the most interesting quality idea on the table right now.

Potential shape:
- latent upscale for global coherence
- pixel-space detail cleanup or learned upscale
- second pass with lighter denoise

This may beat pure latent upscale on blockiness while staying more coherent than a pure RGB round-trip path.

### 4. Subgraph packaging after defaults are clearer

The workflow-first direction still feels right.

But it is better to package a subgraph after:
- branch behavior is better understood
- pass-2 defaults are less experimental

Otherwise we risk freezing a bad UI too early.

## Short version

Where we are now:
- baseline established
- helper nodes work
- no obvious hidden performance penalty versus ReForge
- main cost is pass-2 sampling
- latent-only upscale is not always visually good enough

Where to focus next:
- smarter defaults
- better branch choice
- sampler comparisons
- hybrid strategies
