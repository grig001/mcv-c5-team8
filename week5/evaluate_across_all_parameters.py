import time
import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from diffusers import AutoPipelineForText2Image, DiffusionPipeline


# Model options
model_ids = [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/sdxl-turbo",
    "stabilityai/sd-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0"
]


# Prompts
positive_prompts = [
    "Chocolate lava cake in a minimalist Scandinavian setting, photorealistic, food styling, ultra-detailed"
]
negative_prompts = [
    "blurry, deformed, text, watermark, cartoon"
]

# Parameter ranges (changeable)
cfg_scales = [10, 15]
num_steps = [7, 9]
samplers = ["DDIM", "DDPM"]

# Output directory
output_dir = "task_b_experiments/food/"
os.makedirs(output_dir, exist_ok=True)

# Iterate over models
for model_id in model_ids:
    print(f"Loading model: {model_id}")

    if model_id == "stabilityai/stable-diffusion-2-1":
        print("Running SD 2.1...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    elif model_id == "stabilityai/sd-turbo":
        print("Running SD Turbo...")
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16,
                                                         variant="fp16")

    elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        print("Running SDXL...")
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                 use_safetensors=True, variant="fp16")

    elif model_id == "stabilityai/sdxl-turbo":
        print("Running SDXL Turbo...")
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16,
                                                         variant="fp16")

    pipe = pipe.to("cuda")

    for prompt, neg_prompt in zip(positive_prompts, negative_prompts):
        for sampler in samplers:
            for steps in num_steps:
                for cfg in cfg_scales:
                    # Set scheduler
                    if sampler == "DDIM":
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    elif sampler == "DDPM":
                        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

                    for i in range(2):
                        for prompt_type in ["with_negative", "without_negative"]:
                            current_negative_prompt = neg_prompt if prompt_type == "with_negative" else None
                            start_time = time.time()

                            # Generate image
                            image = pipe(
                                prompt=prompt,
                                negative_prompt=current_negative_prompt,
                                num_inference_steps=steps,
                                guidance_scale=cfg
                            ).images[0]

                            # Save the image
                            filename = f"{model_id.split('/')[-1]}/{prompt_type}/{sampler}/steps{steps}_cfg{cfg}_{prompt_type}_{i}.png"
                            full_path = os.path.join(output_dir, filename)
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            image.save(full_path)

                            print(f"Generated {filename} in {time.time() - start_time:.2f}s")
