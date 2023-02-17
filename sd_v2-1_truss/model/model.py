from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            str(self._data_dir), subfolder="scheduler"
        )
        self._model = StableDiffusionPipeline.from_pretrained(
            str(self._data_dir), scheduler=scheduler, torch_dtype=torch.float16
        )
        self._model.unet.set_use_memory_efficient_attention_xformers(True)
        self._model = self._model.to("cuda")

    def predict(self, request: Dict):
        response = asdict(self._model(**request))
        # Convert to numpy to send back
        response["images"] = [np.asarray(img) for img in response["images"]]
        return response
