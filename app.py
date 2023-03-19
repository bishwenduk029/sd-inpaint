import torch
import cv2
import io
import numpy as np
import random
import imghdr
from pathlib import Path
from typing import Union
from PIL import Image
import sd
import schema
from helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
)
import base64

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model

    torch_dtype = torch.float16
    model = sd.SD15(torch_dtype)

# Inference is ran for every server call
# Reference your preloaded global model variable here.

def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w

def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    print(model_inputs)
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    input = model_inputs.get('files')
    # RGB
    origin_image_bytes = input["image"].read()
    image, alpha_channel, exif = load_img(origin_image_bytes, return_exif=True)

    mask, _ = load_img(input["mask"].read(), gray=True)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    if image.shape[:2] != mask.shape[:2]:
        return (
            f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
            400,
        )

    interpolation = cv2.INTER_CUBIC

    form = model_inputs.get("form")
    size_limit: Union[int, str] = form.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    if "paintByExampleImage" in input:
        paint_by_example_example_image, _ = load_img(
            input["paintByExampleImage"].read()
        )
        paint_by_example_example_image = Image.fromarray(
            paint_by_example_example_image)
    else:
        paint_by_example_example_image = None

    config = schema.Config(
        ldm_steps=form["ldmSteps"],
        ldm_sampler=form["ldmSampler"],
        hd_strategy=form["hdStrategy"],
        zits_wireframe=form["zitsWireframe"],
        hd_strategy_crop_margin=form["hdStrategyCropMargin"],
        hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
        hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
        prompt=form["prompt"],
        negative_prompt=form["negativePrompt"],
        use_croper=form["useCroper"],
        croper_x=form["croperX"],
        croper_y=form["croperY"],
        croper_height=form["croperHeight"],
        croper_width=form["croperWidth"],
        sd_scale=form["sdScale"],
        sd_mask_blur=form["sdMaskBlur"],
        sd_strength=form["sdStrength"],
        sd_steps=form["sdSteps"],
        sd_guidance_scale=form["sdGuidanceScale"],
        sd_sampler=form["sdSampler"],
        sd_seed=form["sdSeed"],
        sd_match_histograms=form["sdMatchHistograms"],
        cv2_flag=form["cv2Flag"],
        cv2_radius=form["cv2Radius"],
        paint_by_example_steps=form["paintByExampleSteps"],
        paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
        paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
        paint_by_example_seed=form["paintByExampleSeed"],
        paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
        paint_by_example_example_image=paint_by_example_example_image,
        p2p_steps=form["p2pSteps"],
        p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
        p2p_guidance_scale=form["p2pGuidanceScale"],
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)
    if config.paint_by_example_seed == -1:
        config.paint_by_example_seed = random.randint(1, 999999999)

    
    image = resize_max_size(image, size_limit=size_limit,
                            interpolation=interpolation)

    mask = resize_max_size(mask, size_limit=size_limit,
                           interpolation=interpolation)


    try:
        res_np_img = model(image, mask, config)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        else:
            return "Internal Server Error", 500
    finally:
        torch.cuda.empty_cache()

    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(origin_image_bytes)

    if exif is not None:
        bytes_io = io.BytesIO(pil_to_bytes(
            Image.fromarray(res_np_img), ext, exif=exif))
    else:
        bytes_io = io.BytesIO(pil_to_bytes(Image.fromarray(res_np_img), ext))

    image.save(bytes_io, format='JPEG')
    image_base64 = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
    return {'image_base64': image_base64}
