from dataclasses import asdict
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.

        self._model = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(self._data_dir), torch_dtype=torch.float16
        )
        self._model = self._model.to("cuda")

    def predict(self, request: Dict):
        if "image" in request:
            request["image"] = Image.fromarray(np.uint8(request["image"])).convert(
                "RGB"
            )
        return asdict(self._model(**request, output_type="numpy"))
