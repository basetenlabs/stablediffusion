import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1-base"


_model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

_model.save_pretrained("./sd_v2-1_truss/data")
