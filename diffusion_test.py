import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionInpaintPipeline

# commandline_args = os.environ.get(
#     "COMMANDLINE_ARGS", "--skip-torch-cuda-test --no-half"
# )


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32, # torch.float16
)
pipe.to("mps")


training_path = Path("data/data_splits/training.txt")


def load_file_list(
    path: Path, image_folder="masked", mask_folder="masks"
) -> List[Tuple[Path, Path]]:
    data_path = path.parent.parent
    img_path = data_path / image_folder
    mask_path = data_path / mask_folder
    with open(path, "r") as f:
        return [
            (
                img_path / (line.strip() + "_stroke_masked.png"),
                mask_path / (line.strip() + "_stroke_mask.png"),
            )
            for line in f.readlines()
        ]


file_list = load_file_list(training_path)

img_path, mask_path = file_list[0]


# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is

image = plt.imread(img_path)
mask_image = plt.imread(mask_path)

prompt = "The face of a cat"
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]


plt.imshow(image)
plt.show()
