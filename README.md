# ComfyUI-GigaHires

Workspace for a new high-quality, high-performance hires-fix / upscale custom node for ComfyUI.

Current status:
- local research notes captured from ComfyUI core and local ReForge source
- helper nodes implemented for the workflow-first direction:
  - `GigaHires Latent Upscale`
  - `GigaHires Image Upscale`
  - `GigaHires Refine Pass`
- coordinator nodes still available:
  - `GigaHires V1`
  - `GigaHires Easy`

Recommended direction:
- use the helper-node workflow
- keep normal pass, upscale, and refine visibly separate
- later collapse that layout into a subgraph with only the important face controls exposed

Helper nodes:
- `GigaHires Latent Upscale`
  - latent-space upscale with `scale_by` or exact target size
- `GigaHires Image Upscale`
  - image-space upscale with `scale_by` or exact target size
  - accepts an `UPSCALE_MODEL` input or internal model dropdown
- `GigaHires Refine Pass`
  - explicit second-pass sampler stage on the already-upscaled latent

Coordinator nodes:
- `GigaHires V1`
  - single post-sampler hires coordinator with most knobs exposed
- `GigaHires Easy`
  - simplified wrapper kept around for quick experiments, not the main UX direction

Primary notes:
- [docs/reforge-hires-fix-notes.md](docs/reforge-hires-fix-notes.md)
- [docs/explicit-two-pass-reference.md](docs/explicit-two-pass-reference.md)
- [docs/helper-nodes-subgraph-plan.md](docs/helper-nodes-subgraph-plan.md)
- [docs/baseline-and-next-steps.md](docs/baseline-and-next-steps.md)
- [example_workflows/explicit_two_pass_reference.json](/mnt/d/comfyui/custom_nodes/ComfyUI-GigaHires/example_workflows/explicit_two_pass_reference.json)
- [example_workflows/helper_nodes_reference.json](/mnt/d/comfyui/custom_nodes/ComfyUI-GigaHires/example_workflows/helper_nodes_reference.json)
- [reforge repo](/mnt/d/stable-diffusion-webui-reForge)
