from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        self._model = StableDiffusionInpaintPipeline.from_pretrained(
            self._data_dir, torch_dtype=torch.float16
        )
        self._model = self._model.to("cuda")

    def predict(self, request: Dict):
        if "image" in request:
            request["image"] = Image.fromarray(request["image"], "RGB")

        if "mask_image" in request:
            request["mask_image"] = Image.fromarray(request["mask_image"], "RGB")

        response = self._model(**request)
        # Convert to numpy to send back
        response.images = [np.asarray(img) for img in response.images]
        return asdict(response)
