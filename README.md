# ComfyUI-GigaHires

Workspace for a new high-quality, high-performance hires-fix / upscale custom node for ComfyUI.

Current status:
- local research notes captured from ComfyUI core and local ReForge source
- first runnable node implemented: `GigaHires V1`

Implemented in v1:
- post-sampler hires-fix coordinator
- accepts a first-pass latent
- latent branch with ReForge-style interpolation modes
- learned upscaler branch with optional `UPSCALE_MODEL` input or internal selector
- second-pass refinement using Comfy's sampling helper
- optional pass-2 conditioning overrides
- regular or tiled VAE mode for pixel-path work
- debug outputs for pre-refine and post-refine comparison

Node outputs:
- `pass2_latent`
- `pass2_image`
- `refined_latent`
- `refined_image`
- `debug_info`

Suggested first test:
- generate a base latent with your usual `KSampler` or `SamplerCustom`
- feed the main latent output into `GigaHires V1`
- start with:
  - `branch_mode = latent`
  - `latent_mode = Latent (antialiased)`
  - `sizing_mode = scale`
  - `scale_by = 2.0`
  - `steps = 12`
  - `denoise = 0.35`
- compare `pass2_image` vs `refined_image`
- then try `branch_mode = upscale_model` with a loaded upscale model

Primary notes:
- [docs/reforge-hires-fix-notes.md](docs/reforge-hires-fix-notes.md)
- [reforge repo](/mnt/d/stable-diffusion-webui-reForge)
