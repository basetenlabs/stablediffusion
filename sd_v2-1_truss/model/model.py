from dataclasses import asdict
from typing import Dict, List

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.

        self._model = StableDiffusionPipeline.from_pretrained(
            str(self._data_dir), torch_dtype=torch.float16
        )
        self._model.scheduler = DPMSolverMultistepScheduler.from_config(
            self._model.scheduler.config
        )
        self._model = self._model.to("cuda")

    def predict(self, request: Dict):
        return asdict(self._model(**request, output_type="numpy"))
