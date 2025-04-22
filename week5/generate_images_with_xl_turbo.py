import torch
from pathlib import Path
import pandas as pd
from diffusers import AutoPipelineForText2Image, DDPMScheduler
import os


# Load prompt and image name from csv
df = pd.read_csv("prompts/dataset_likely_last_3240.csv")


# SDXL Turbo
print("Running SDXL Turbo...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

print("Running with default parameters...")


# Default parameters
output_dir = Path("generated_images/dataset_likely/default")
os.makedirs(output_dir, exist_ok=True)

for _, row in df.iterrows():

    image = pipe(prompt=row["prompt"], num_inference_steps=1, guidance_scale=0.0).images[0]
    filename = row["image_name"]
    image.save(output_dir / filename)

print("Running with optimized parameters...")

# Optimized parameters
output_dir = Path("generated_images/dataset_likely/optimized")
os.makedirs(output_dir, exist_ok=True)

negative_prompt = "blurry, deformed, text, watermark, cartoon"
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)


for _, row in df.iterrows():

    image = pipe(
        prompt=row["prompt"],
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=9
    ).images[0]

    filename = row["image_name"]
    image.save(output_dir / filename)

print("All Images saved.")
