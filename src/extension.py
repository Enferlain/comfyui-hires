from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
from typing_extensions import override


LATENT_MODE_LABELS = [
    "Latent",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Latent (nearest)",
    "Latent (nearest-exact)",
]

LATENT_MODE_MAP = {
    "Latent": ("bilinear", False),
    "Latent (antialiased)": ("bilinear", True),
    "Latent (bicubic)": ("bicubic", False),
    "Latent (bicubic antialiased)": ("bicubic", True),
    "Latent (nearest)": ("nearest", False),
    "Latent (nearest-exact)": ("nearest-exact", False),
}


def _flatten_decoded(images: torch.Tensor) -> torch.Tensor:
    if len(images.shape) == 5:
        return images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _get_spatial_compression(vae) -> int:
    compression = vae.spacial_compression_encode()
    if isinstance(compression, tuple):
        compression = compression[-1]
    return int(compression)


def _snap_size(value: int, compression: int) -> int:
    return max(compression, int(round(value / compression)) * compression)


def _resolve_target_size(
    latent: dict,
    vae,
    sizing_mode: str,
    scale_by: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int, int]:
    compression = _get_spatial_compression(vae)
    latent_samples = latent["samples"]
    base_width = latent_samples.shape[-1] * compression
    base_height = latent_samples.shape[-2] * compression

    if sizing_mode == "scale":
        requested_width = max(1, round(base_width * scale_by))
        requested_height = max(1, round(base_height * scale_by))
    else:
        if width == 0 and height == 0:
            requested_width = base_width
            requested_height = base_height
        elif width == 0:
            requested_height = max(1, height)
            requested_width = max(1, round(base_width * requested_height / base_height))
        elif height == 0:
            requested_width = max(1, width)
            requested_height = max(1, round(base_height * requested_width / base_width))
        else:
            requested_width = max(1, width)
            requested_height = max(1, height)

    target_width = _snap_size(requested_width, compression)
    target_height = _snap_size(requested_height, compression)
    return base_width, base_height, target_width, target_height, compression


def _resize_noise_mask(noise_mask, target_height: int, target_width: int):
    if getattr(noise_mask, "is_nested", False):
        return noise_mask

    mask = noise_mask
    squeeze_channel = False
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
        squeeze_channel = True
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
        squeeze_channel = True

    resized = F.interpolate(mask, size=(target_height, target_width), mode="nearest-exact")
    if squeeze_channel:
        resized = resized.squeeze(1)
    return resized


def _copy_latent_with_samples(source_latent: dict, samples: torch.Tensor) -> dict:
    out = source_latent.copy()
    out["samples"] = samples
    if "noise_mask" in source_latent:
        out["noise_mask"] = _resize_noise_mask(source_latent["noise_mask"], samples.shape[-2], samples.shape[-1])
    return out


def _resize_image_tensor(image: torch.Tensor, width: int, height: int, method: str = "lanczos") -> torch.Tensor:
    if image.shape[1] == height and image.shape[2] == width:
        return image
    return comfy.utils.common_upscale(
        image.movedim(-1, 1),
        width,
        height,
        method,
        "disabled",
    ).movedim(1, -1)


def _decode_latent(vae, latent: dict, vae_mode: str, tile_size: int, overlap: int) -> torch.Tensor:
    samples = latent["samples"]
    if vae_mode == "tiled":
        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            samples,
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
        )
    else:
        images = vae.decode(samples)
    return _flatten_decoded(images)


def _encode_image(vae, image: torch.Tensor, vae_mode: str, tile_size: int, overlap: int) -> dict:
    pixels = image[..., :3]
    if vae_mode == "tiled":
        samples = vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
    else:
        samples = vae.encode(pixels)
    return {"samples": samples}


def _upscale_latent_tensor(samples: torch.Tensor, width: int, height: int, latent_mode: str) -> torch.Tensor:
    mode, antialias = LATENT_MODE_MAP[latent_mode]
    interpolate_kwargs = {"size": (height, width), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        interpolate_kwargs["antialias"] = antialias
        interpolate_kwargs["align_corners"] = False
    return F.interpolate(samples, **interpolate_kwargs)


def _load_upscale_model(upscale_model, upscale_model_name: str):
    if upscale_model is not None:
        return upscale_model, getattr(upscale_model, "model_name", None) or "<input>"
    if upscale_model_name == "None":
        raise ValueError("branch_mode 'upscale_model' requires either an 'upscale_model' input or a selected 'upscale_model_name'.")
    loaded = UpscaleModelLoader.execute(upscale_model_name)[0]
    return loaded, upscale_model_name


class GigaHiresV1(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GigaHiresV1",
            display_name="GigaHires V1",
            category="sampling/upscale",
            description="Post-sampler hires-fix coordinator. Takes a first-pass latent, upscales it using a latent or learned-model path, then runs a second refinement pass.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Combo.Input("branch_mode", options=["latent", "upscale_model"], default="latent"),
                io.Combo.Input("latent_mode", options=LATENT_MODE_LABELS, default="Latent (antialiased)"),
                io.Combo.Input("sizing_mode", options=["scale", "target"], default="scale"),
                io.Float.Input("scale_by", default=2.0, min=0.1, max=8.0, step=0.01),
                io.Int.Input("target_width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("target_height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Combo.Input("upscale_model_name", options=["None", *folder_paths.get_filename_list("upscale_models")], default="None"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=12, min=1, max=10000),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Float.Input("denoise", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("vae_mode", options=["regular", "tiled"], default="regular"),
                io.Int.Input("vae_tile_size", default=512, min=64, max=4096, step=64),
                io.Int.Input("vae_overlap", default=64, min=0, max=4096, step=32),
                io.Conditioning.Input("positive_pass2", optional=True),
                io.Conditioning.Input("negative_pass2", optional=True),
                io.UpscaleModel.Input("upscale_model", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="pass2_latent"),
                io.Image.Output(display_name="pass2_image"),
                io.Latent.Output(display_name="refined_latent"),
                io.Image.Output(display_name="refined_image"),
                io.String.Output(display_name="debug_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        vae,
        positive,
        negative,
        latent,
        branch_mode,
        latent_mode,
        sizing_mode,
        scale_by,
        target_width,
        target_height,
        upscale_model_name,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        vae_mode,
        vae_tile_size,
        vae_overlap,
        positive_pass2=None,
        negative_pass2=None,
        upscale_model=None,
    ) -> io.NodeOutput:
        start_time = time.perf_counter()

        positive_2 = positive_pass2 if positive_pass2 is not None else positive
        negative_2 = negative_pass2 if negative_pass2 is not None else negative

        base_width, base_height, resolved_width, resolved_height, compression = _resolve_target_size(
            latent,
            vae,
            sizing_mode,
            scale_by,
            target_width,
            target_height,
        )

        target_latent_width = resolved_width // compression
        target_latent_height = resolved_height // compression

        branch_start = time.perf_counter()
        used_upscale_model_name = None

        if branch_mode == "latent":
            pass2_latent = _copy_latent_with_samples(
                latent,
                _upscale_latent_tensor(latent["samples"], target_latent_width, target_latent_height, latent_mode),
            )
            pass2_image = _decode_latent(vae, pass2_latent, vae_mode, vae_tile_size, vae_overlap)
        else:
            decoded = _decode_latent(vae, latent, vae_mode, vae_tile_size, vae_overlap)
            decoded = _resize_image_tensor(decoded, base_width, base_height, "lanczos")

            if resolved_width > base_width or resolved_height > base_height:
                loaded_upscale_model, used_upscale_model_name = _load_upscale_model(upscale_model, upscale_model_name)
                upscaled = ImageUpscaleWithModel.execute(loaded_upscale_model, decoded)[0]
            else:
                upscaled = decoded
                used_upscale_model_name = "<skipped-downscale>"

            pass2_image = _resize_image_tensor(upscaled, resolved_width, resolved_height, "lanczos")
            pass2_latent = _copy_latent_with_samples(
                latent,
                _encode_image(vae, pass2_image, vae_mode, vae_tile_size, vae_overlap)["samples"],
            )

        upscale_elapsed_ms = round((time.perf_counter() - branch_start) * 1000.0, 2)

        sample_start = time.perf_counter()
        refined_latent = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive_2,
            negative_2,
            pass2_latent,
            denoise=denoise,
        )[0]
        sampling_elapsed_ms = round((time.perf_counter() - sample_start) * 1000.0, 2)

        decode_start = time.perf_counter()
        refined_image = _decode_latent(vae, refined_latent, vae_mode, vae_tile_size, vae_overlap)
        decode_elapsed_ms = round((time.perf_counter() - decode_start) * 1000.0, 2)

        debug_info = json.dumps(
            {
                "branch_mode": branch_mode,
                "latent_mode": latent_mode if branch_mode == "latent" else None,
                "used_upscale_model_name": used_upscale_model_name,
                "sizing_mode": sizing_mode,
                "base_size": [base_width, base_height],
                "resolved_size": [resolved_width, resolved_height],
                "compression": compression,
                "pass2_latent_shape": list(pass2_latent["samples"].shape),
                "refined_latent_shape": list(refined_latent["samples"].shape),
                "pass2_conditioning_overridden": positive_pass2 is not None or negative_pass2 is not None,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
                "vae_mode": vae_mode,
                "timings_ms": {
                    "upscale_prepare": upscale_elapsed_ms,
                    "refine_sample": sampling_elapsed_ms,
                    "final_decode": decode_elapsed_ms,
                    "total": round((time.perf_counter() - start_time) * 1000.0, 2),
                },
            },
            indent=2,
        )

        return io.NodeOutput(pass2_latent, pass2_image, refined_latent, refined_image, debug_info)


class GigaHiresExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GigaHiresV1,
        ]


async def comfy_entrypoint() -> GigaHiresExtension:
    return GigaHiresExtension()
