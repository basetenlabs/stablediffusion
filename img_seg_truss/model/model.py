from typing import Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from transformers import (  # Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    AutoProcessor,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
)

transform = T.ToPILImage()

MODEL_NAME = "facebook/maskformer-swin-base-ade"
MODEL_TWO_NAME = "facebook/mask2former-swin-base-coco-panoptic"


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._processor = None
        self._model = None

    def load(self):
        # # Load model here and assign to self._model.
        # self._processor = MaskFormerImageProcessor.from_pretrained(MODEL_NAME)
        # self._model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_NAME)
        # # Load model here and assign to self._model.
        # self._processor = AutoImageProcessor.from_pretrained(MODEL_TWO_NAME)
        self._processor = AutoProcessor.from_pretrained(
            "shi-labs/oneformer_coco_swin_large"
        )
        self._model = AutoModelForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_coco_swin_large"
        )
        # = Mask2FormerForUniversalSegmentation.from_pretrained(
        #     MODEL_TWO_NAME
        # )

        # TODO: uncomment for GPU support
        # self._model = self._model.to("cuda")

    def predict(self, request: Dict) -> Dict[str, List]:
        image = request["image"]
        # Do np to PIL conversion
        image = Image.fromarray(request["image"], "RGB")
        # Pre-process
        inputs = self._processor(image, task_inputs=["panoptic"], return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        panoptic_segmentation = self._processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]  # , label_ids_to_fuse=[0]
        )[0]

        for segment in panoptic_segmentation["segments_info"]:
            print(segment)
            segment_label_id = segment["label_id"]
            segment["label"] = self._model.config.id2label[segment_label_id]
        panoptic_segmentation["segmentation"] = (
            panoptic_segmentation["segmentation"].numpy().astype(np.uint8)
        )

        # for segment in results["segments_info"]:
        #     print(segment)
        # seg_img = transform(torch.nn.Softmax(dim=0)(semantic_segmentation.float()))
        # seg_img = seg_img.point(lambda p: p > 128 and 255)
        # seg_img = seg_img.point(lambda x: 0 if x < 5 else 255, "1")
        # seg_img = seg_img.resize((768, 768))
        # seg_img = seg_img.convert("1")
        return panoptic_segmentation
