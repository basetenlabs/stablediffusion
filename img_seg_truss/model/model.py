from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoProcessor


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._processor = None
        self._model = None

    def load(self):
        # # Load model here and assign to self._model.
        self._processor = AutoProcessor.from_pretrained(
            "shi-labs/oneformer_coco_swin_large"
        )
        self._model = AutoModelForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_coco_swin_large"
        ).to("cuda")

    def predict(self, request: Dict) -> Dict[str, List]:
        image = request["image"]
        # Do np to PIL conversion
        image = Image.fromarray(np.asarray(image), "RGB")

        # Pre-process
        inputs = self._processor(
            image, task_inputs=["panoptic"], return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = self._model(**inputs)

        panoptic_segmentation = self._processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]  # , label_ids_to_fuse=[0]
        )[0]

        for segment in panoptic_segmentation["segments_info"]:
            segment_label_id = segment["label_id"]
            segment["label"] = self._model.config.id2label[segment_label_id]

        panoptic_segmentation["segmentation"] = (
            panoptic_segmentation["segmentation"].cpu().numpy()
        )

        return panoptic_segmentation
