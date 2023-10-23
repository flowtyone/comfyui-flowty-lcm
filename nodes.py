from .lcm.lcm_scheduler import LCMScheduler
from .lcm.lcm_pipeline import LatentConsistencyModelPipeline
from os import path
import time
import torch
import random
import numpy as np
from comfy.model_management import get_torch_device

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

class LCMSampler:

    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "size": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                     "positive_prompt": ("STRING", {"multiline": True}),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, seed, steps, cfg, positive_prompt, size, num_images):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                local_files_only=True,
                scheduler=self.scheduler
            )
            self.pipe.to(get_torch_device())

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            prompt=positive_prompt,
            width=size,
            height=size,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=4,
            output_type="np",
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)

NODE_CLASS_MAPPINGS = {
    "LCMSampler": LCMSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCMSampler" : "LCMSampler"
}