import PIL
import PIL.Image
import cv2
import torch

from base import DiffusionInpaintModel
from utils import set_seed
from schema import Config


class PaintByExample(DiffusionInpaintModel):
    name = "paint_by_example"
    pad_mod = 8
    min_size = 512

    def init_model(self, model, **kwargs):
        self.model = model

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        output = self.model(
            image=PIL.Image.fromarray(image),
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            example_image=config.paint_by_example_example_image,
            num_inference_steps=config.paint_by_example_steps,
            output_type='np.array',
            generator=torch.manual_seed(config.paint_by_example_seed)
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def forward_post_process(self, result, image, mask, config):
        if config.paint_by_example_match_histograms:
            result = self._match_histograms(result, image[:, :, ::-1], mask)

        if config.paint_by_example_mask_blur != 0:
            k = 2 * config.paint_by_example_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return result, image, mask

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True
