import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1-base"
StableDiffusionPipeline.from_pretrained(
    # str(self._data_dir),
    model_id,
    scheduler=EulerDiscreteScheduler.from_pretrained(
        # str(self._data_dir),
        model_id,
        subfolder="scheduler",
    ),
    torch_dtype=torch.float16,
).save_pretrained("./difussion_2-1/data")
