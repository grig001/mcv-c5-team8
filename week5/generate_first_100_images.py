import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import os
import gc

# Load prompts
with open("prompts/first_100.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Output directory setup
base_dir = Path("generated_images/first_100")
model_dirs = {
    "sd_2_1": base_dir / "sd_2_1",
    "sd_turbo": base_dir / "sd_turbo",
    "sdxl": base_dir / "sdxl",
    "sdxl_turbo": base_dir / "sdxl_turbo",
}

for dir_path in model_dirs.values():
    os.makedirs(dir_path, exist_ok=True)


def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()


# SD 2.1
print("Running SD 2.1...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(model_dirs["sd_2_1"] / f"{i:03d}.png")

del pipe
clear_cuda()

# SD Turbo
print("Running SD Turbo...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save(model_dirs["sd_turbo"] / f"{i:03d}.png")

del pipe
clear_cuda()

# SDXL
print("Running SDXL...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt).images[0]
    image.save(model_dirs["sdxl"] / f"{i:03d}.png")

del pipe
clear_cuda()

# SDXL Turbo
print("Running SDXL Turbo...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save(model_dirs["sdxl_turbo"] / f"{i:03d}.png")

del pipe
clear_cuda()

print("All models done. Images saved.")
