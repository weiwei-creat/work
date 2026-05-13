# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from constants import MATFUSE_ROOT_DIR
import os
import importlib.util
import sys
import random
from PIL import Image

_generate_module = None


def _matfuse_disabled():
    return os.environ.get("SAGE_DISABLE_MATFUSE", "0").lower() in {"1", "true", "yes", "on"}


def _placeholder_texture(color=(192, 192, 192)):
    return Image.new("RGB", (512, 512), color)


def _load_generate_module():
    global _generate_module
    if _generate_module is None:
        generate_py_path = os.path.join(MATFUSE_ROOT_DIR, "generate.py")
        spec = importlib.util.spec_from_file_location("generate_module", generate_py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generate_module"] = module
        spec.loader.exec_module(module)
        _generate_module = module
    return _generate_module


def generate_texture_map_from_prompt(prompt):
    if _matfuse_disabled():
        return _placeholder_texture()
    return _load_generate_module().generate_texture_map_from_prompt(prompt)


def generate_texture_map_from_prompt_and_sketch(prompt, sketch):
    if _matfuse_disabled():
        return _placeholder_texture()
    return _load_generate_module().generate_texture_map_from_prompt_and_sketch(prompt, sketch)


def generate_texture_map_from_prompt_and_sketch_and_image(prompt, sketch, image):
    if _matfuse_disabled():
        return _placeholder_texture()
    return _load_generate_module().generate_texture_map_from_prompt_and_sketch_and_image(prompt, sketch, image)


def generate_texture_map_from_prompt_and_color(prompt, color):
    if _matfuse_disabled():
        rgb = tuple(int(max(0, min(1, c)) * 255) for c in color[:3])
        return _placeholder_texture(rgb)
    return _load_generate_module().generate_texture_map_from_prompt_and_color(prompt, color)


def generate_texture_map_from_prompt_and_color_and_sketch(prompt, color, sketch):
    if _matfuse_disabled():
        rgb = tuple(int(max(0, min(1, c)) * 255) for c in color[:3])
        return _placeholder_texture(rgb)
    return _load_generate_module().generate_texture_map_from_prompt_and_color_and_sketch(prompt, color, sketch)


def generate_texture_map_from_prompt_and_color_palette(prompt, color_palette):
    if _matfuse_disabled():
        if color_palette:
            color = color_palette[0]
            rgb = tuple(int(max(0, min(1, c)) * 255) for c in color[:3])
        else:
            rgb = (192, 192, 192)
        return _placeholder_texture(rgb)
    return _load_generate_module().generate_texture_map_from_prompt_and_color_palette(prompt, color_palette)

def material_generate_from_prompt(prompts):
    results = []
    for prompt in prompts:
        texture_map_pil = generate_texture_map_from_prompt(prompt)

        results.append(texture_map_pil)
    return results

# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
# from diffusers.utils import load_image
# import numpy as np
# import torch

# import cv2
# from PIL import Image

# # initialize the models and pipeline
# controlnet_conditioning_scale = 0.5  # recommended for good generalization
# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
# )
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
# )
# pipe.enable_model_cpu_offload()


# def generate_texture_map_from_prompt_and_sketch_controlnet(prompt, sketch):

#     sketch = Image.fromarray(sketch)

#     # Set a fixed seed for reproducible results
#     generator = torch.Generator().manual_seed(random.randint(0, 1000000))

#     # generate image with stable parameters for architectural materials
#     image = pipe(
#         prompt, 
#         image=sketch,
#         height=512,
#         width=512,
#         num_inference_steps=30,  # More steps for better quality and stability
#         guidance_scale=7.5,  # Higher guidance for better prompt adherence
#         controlnet_conditioning_scale=controlnet_conditioning_scale,
#         control_guidance_start=0.0,  # Apply control from beginning
#         control_guidance_end=0.8,  # Reduce control influence near end for natural results
#         generator=generator,  # Fixed seed for consistency
#         eta=0.0,  # Deterministic sampling for stability
#         negative_prompt="low quality, bad quality, dirty, smudged, stained, grimy, dusty, fingerprints, water spots, streaks, cloudy, foggy, cracked, broken, distorted, warped, blurry, scratched, damaged, weathered, aged, yellowed, tinted, reflective glare, harsh reflections, unrealistic, fantasy elements, cartoon, anime, abstract patterns, decorative ornaments, "
#     ).images[0]

#     return image
