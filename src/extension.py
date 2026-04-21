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

    target_width, target_height = _resolve_target_size_from_dims(
        base_width,
        base_height,
        sizing_mode,
        scale_by,
        width,
        height,
        compression,
    )
    return base_width, base_height, target_width, target_height, compression


def _resolve_target_size_from_dims(
    base_width: int,
    base_height: int,
    sizing_mode: str,
    scale_by: float,
    width: int,
    height: int,
    snap_multiple: int = 1,
) -> tuple[int, int]:
    snap_multiple = max(1, snap_multiple)

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

    target_width = _snap_size(requested_width, snap_multiple)
    target_height = _snap_size(requested_height, snap_multiple)
    return target_width, target_height


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
        return upscale_model, getattr(upscale_model, "model_name", None) or "<input>", False
    if upscale_model_name == "None":
        raise ValueError("branch_mode 'upscale_model' requires either an 'upscale_model' input or a selected 'upscale_model_name'.")
    loaded = UpscaleModelLoader.execute(upscale_model_name)[0]
    return loaded, upscale_model_name, True


def _run_image_upscale(
    image: torch.Tensor,
    method: str,
    sizing_mode: str,
    scale_by: float,
    target_width: int,
    target_height: int,
    upscale_model_name: str = "None",
    upscale_model=None,
):
    start_time = time.perf_counter()
    base_width = image.shape[2]
    base_height = image.shape[1]
    resolved_width, resolved_height = _resolve_target_size_from_dims(
        base_width,
        base_height,
        sizing_mode,
        scale_by,
        target_width,
        target_height,
    )

    used_upscale_model_name = None
    loaded_internally = False
    upscale_model_elapsed_ms = 0.0
    resize_elapsed_ms = 0.0
    if method == "upscale_model":
        if resolved_width > base_width or resolved_height > base_height:
            loaded_upscale_model, used_upscale_model_name, loaded_internally = _load_upscale_model(upscale_model, upscale_model_name)
            upscale_model_start = time.perf_counter()
            upscaled = ImageUpscaleWithModel.execute(loaded_upscale_model, image)[0]
            upscale_model_elapsed_ms = round((time.perf_counter() - upscale_model_start) * 1000.0, 2)
        else:
            upscaled = image
            used_upscale_model_name = "<skipped-downscale>"
        resize_start = time.perf_counter()
        final_image = _resize_image_tensor(upscaled, resolved_width, resolved_height, "lanczos")
        resize_elapsed_ms = round((time.perf_counter() - resize_start) * 1000.0, 2)
    else:
        resize_start = time.perf_counter()
        final_image = _resize_image_tensor(image, resolved_width, resolved_height, "lanczos")
        resize_elapsed_ms = round((time.perf_counter() - resize_start) * 1000.0, 2)

    return final_image, {
        "method": method,
        "used_upscale_model_name": used_upscale_model_name,
        "upscale_model_loaded_internally": loaded_internally,
        "base_size": [base_width, base_height],
        "resolved_size": [resolved_width, resolved_height],
        "timings_ms": {
            "upscale_model": upscale_model_elapsed_ms,
            "resize": resize_elapsed_ms,
            "total": round((time.perf_counter() - start_time) * 1000.0, 2),
        },
    }


def _run_giga_hires(
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
    extra_debug=None,
):
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
    loaded_internally = False

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
            loaded_upscale_model, used_upscale_model_name, loaded_internally = _load_upscale_model(upscale_model, upscale_model_name)
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

    debug_payload = {
        "branch_mode": branch_mode,
        "latent_mode": latent_mode if branch_mode == "latent" else None,
        "used_upscale_model_name": used_upscale_model_name,
        "upscale_model_loaded_internally": loaded_internally,
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
    }
    if extra_debug:
        debug_payload.update(extra_debug)

    return pass2_latent, pass2_image, refined_latent, refined_image, json.dumps(debug_payload, indent=2)


def _easy_settings(quality: str, detail: str):
    detail_presets = {
        "subtle": {"denoise": 0.22, "cfg": 6.5},
        "balanced": {"denoise": 0.35, "cfg": 7.0},
        "strong": {"denoise": 0.5, "cfg": 7.5},
    }
    quality_presets = {
        "fast": {"steps": 8, "sampler_name": "euler", "scheduler": "normal", "vae_mode": "regular"},
        "balanced": {"steps": 12, "sampler_name": "dpmpp_2m", "scheduler": "karras", "vae_mode": "regular"},
        "high": {"steps": 16, "sampler_name": "dpmpp_2m", "scheduler": "karras", "vae_mode": "tiled"},
    }
    return {**detail_presets[detail], **quality_presets[quality], "latent_mode": "Latent (antialiased)"}


class GigaHiresLatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GigaHiresLatentUpscale",
            display_name="GigaHires Latent Upscale",
            category="sampling/upscale",
            description="Upscales a latent with explicit sizing controls so it can feed a visible second-pass refine stage.",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Combo.Input("latent_mode", options=LATENT_MODE_LABELS, default="Latent (antialiased)"),
                io.Combo.Input("sizing_mode", options=["scale", "target"], default="scale"),
                io.Float.Input("scale_by", default=2.0, min=0.1, max=8.0, step=0.01),
                io.Int.Input("target_width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("target_height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
            ],
            outputs=[
                io.Latent.Output(display_name="upscaled_latent"),
                io.String.Output(display_name="debug_info"),
            ],
        )

    @classmethod
    def execute(cls, latent, vae, latent_mode, sizing_mode, scale_by, target_width, target_height) -> io.NodeOutput:
        start_time = time.perf_counter()
        base_width, base_height, resolved_width, resolved_height, compression = _resolve_target_size(
            latent,
            vae,
            sizing_mode,
            scale_by,
            target_width,
            target_height,
        )
        upscaled = _copy_latent_with_samples(
            latent,
            _upscale_latent_tensor(
                latent["samples"],
                resolved_width // compression,
                resolved_height // compression,
                latent_mode,
            ),
        )
        debug_info = json.dumps(
            {
                "node": "GigaHiresLatentUpscale",
                "latent_mode": latent_mode,
                "sizing_mode": sizing_mode,
                "base_size": [base_width, base_height],
                "resolved_size": [resolved_width, resolved_height],
                "latent_shape": list(upscaled["samples"].shape),
                "timings_ms": {
                    "total": round((time.perf_counter() - start_time) * 1000.0, 2),
                },
            },
            indent=2,
        )
        return io.NodeOutput(upscaled, debug_info)


class GigaHiresImageUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GigaHiresImageUpscale",
            display_name="GigaHires Image Upscale",
            category="sampling/upscale",
            description="Upscales an image with either plain resize or a learned upscaler, with explicit scale-by or target-size controls.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("method", options=["upscale_model", "lanczos"], default="upscale_model"),
                io.Combo.Input("sizing_mode", options=["scale", "target"], default="scale"),
                io.Float.Input("scale_by", default=2.0, min=0.1, max=8.0, step=0.01),
                io.Int.Input("target_width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("target_height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Combo.Input("upscale_model_name", options=["None", *folder_paths.get_filename_list("upscale_models")], default="None"),
                io.UpscaleModel.Input("upscale_model", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="upscaled_image"),
                io.String.Output(display_name="debug_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        method,
        sizing_mode,
        scale_by,
        target_width,
        target_height,
        upscale_model_name,
        upscale_model=None,
    ) -> io.NodeOutput:
        upscaled, debug = _run_image_upscale(
            image=image,
            method=method,
            sizing_mode=sizing_mode,
            scale_by=scale_by,
            target_width=target_width,
            target_height=target_height,
            upscale_model_name=upscale_model_name,
            upscale_model=upscale_model,
        )
        debug["node"] = "GigaHiresImageUpscale"
        return io.NodeOutput(upscaled, json.dumps(debug, indent=2))


class GigaHiresRefinePass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GigaHiresRefinePass",
            display_name="GigaHires Refine Pass",
            category="sampling/upscale",
            description="Runs the visible second-pass refinement step on an already-upscaled latent.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=12, min=1, max=10000),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="dpmpp_2m"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="karras"),
                io.Float.Input("denoise", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("vae_mode", options=["regular", "tiled"], default="regular"),
                io.Int.Input("vae_tile_size", default=512, min=64, max=4096, step=64),
                io.Int.Input("vae_overlap", default=64, min=0, max=4096, step=32),
            ],
            outputs=[
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
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        vae_mode,
        vae_tile_size,
        vae_overlap,
    ) -> io.NodeOutput:
        start_time = time.perf_counter()
        sample_start = time.perf_counter()
        refined_latent = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]
        sample_elapsed_ms = round((time.perf_counter() - sample_start) * 1000.0, 2)
        decode_start = time.perf_counter()
        refined_image = _decode_latent(vae, refined_latent, vae_mode, vae_tile_size, vae_overlap)
        decode_elapsed_ms = round((time.perf_counter() - decode_start) * 1000.0, 2)
        debug_info = json.dumps(
            {
                "node": "GigaHiresRefinePass",
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "vae_mode": vae_mode,
                "latent_shape": list(refined_latent["samples"].shape),
                "timings_ms": {
                    "sample": sample_elapsed_ms,
                    "decode": decode_elapsed_ms,
                    "total": round((time.perf_counter() - start_time) * 1000.0, 2),
                },
            },
            indent=2,
        )
        return io.NodeOutput(refined_latent, refined_image, debug_info)


class GigaHiresDebugPrint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GigaHiresDebugPrint",
            display_name="GigaHires Debug Print",
            category="utils",
            description="Prints a STRING value to the ComfyUI console and passes it through unchanged.",
            inputs=[
                io.String.Input("text", multiline=True),
                io.String.Input("label", default="GigaHires Debug"),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, text, label) -> io.NodeOutput:
        print(f"[{label}]")
        print(text)
        return io.NodeOutput(text)


class GigaHiresEasy(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GigaHiresEasy",
            display_name="GigaHires Easy",
            category="sampling/upscale",
            description="Simplified hires-fix wrapper. Feed it your first-pass latent and choose a scale, quality, and detail level.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Float.Input("scale_by", default=2.0, min=1.0, max=8.0, step=0.05),
                io.Combo.Input("mode", options=["latent", "upscale_model"], default="latent"),
                io.Combo.Input("quality", options=["fast", "balanced", "high"], default="balanced"),
                io.Combo.Input("detail", options=["subtle", "balanced", "strong"], default="balanced"),
                io.Combo.Input("upscale_model_name", options=["None", *folder_paths.get_filename_list("upscale_models")], default="None"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.UpscaleModel.Input("upscale_model", optional=True),
            ],
            outputs=[
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
        scale_by,
        mode,
        quality,
        detail,
        upscale_model_name,
        seed,
        upscale_model=None,
    ) -> io.NodeOutput:
        presets = _easy_settings(quality, detail)
        branch_mode = "upscale_model" if mode == "upscale_model" else "latent"
        pass2_latent, pass2_image, refined_latent, refined_image, debug_info = _run_giga_hires(
            model=model,
            vae=vae,
            positive=positive,
            negative=negative,
            latent=latent,
            branch_mode=branch_mode,
            latent_mode=presets["latent_mode"],
            sizing_mode="scale",
            scale_by=scale_by,
            target_width=0,
            target_height=0,
            upscale_model_name=upscale_model_name,
            seed=seed,
            steps=presets["steps"],
            cfg=presets["cfg"],
            sampler_name=presets["sampler_name"],
            scheduler=presets["scheduler"],
            denoise=presets["denoise"],
            vae_mode=presets["vae_mode"],
            vae_tile_size=512,
            vae_overlap=64,
            upscale_model=upscale_model,
            extra_debug={
                "easy_mode": True,
                "easy_quality": quality,
                "easy_detail": detail,
                "easy_pass2_image_shape": list(pass2_image.shape),
            },
        )
        return io.NodeOutput(refined_latent, refined_image, debug_info)


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
        pass2_latent, pass2_image, refined_latent, refined_image, debug_info = _run_giga_hires(
            model=model,
            vae=vae,
            positive=positive,
            negative=negative,
            latent=latent,
            branch_mode=branch_mode,
            latent_mode=latent_mode,
            sizing_mode=sizing_mode,
            scale_by=scale_by,
            target_width=target_width,
            target_height=target_height,
            upscale_model_name=upscale_model_name,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            vae_mode=vae_mode,
            vae_tile_size=vae_tile_size,
            vae_overlap=vae_overlap,
            positive_pass2=positive_pass2,
            negative_pass2=negative_pass2,
            upscale_model=upscale_model,
        )
        return io.NodeOutput(pass2_latent, pass2_image, refined_latent, refined_image, debug_info)


class GigaHiresExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GigaHiresLatentUpscale,
            GigaHiresImageUpscale,
            GigaHiresRefinePass,
            GigaHiresDebugPrint,
            GigaHiresEasy,
            GigaHiresV1,
        ]


async def comfy_entrypoint() -> GigaHiresExtension:
    return GigaHiresExtension()
