
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from diffusers.utils.logging import disable_progress_bar

from tqdm import tqdm

from diffusers import DDIMScheduler
from diffusers import DPMSolverMultistepScheduler

disable_progress_bar()

def load_file_list(
    path: Path,
) -> List[Tuple[Path, Path ,Path]]:
    data_path = path.parent.parent
    original_path = data_path / "originals"
    masked_path = data_path / "masked"
    mask_path = data_path / "masks"

    with open(path, "r") as f:
        return [
            (
                original_path / (line.strip() + ".jpg"),
                masked_path / (line.strip() + "_stroke_masked.png"),
                mask_path / (line.strip() + "_stroke_mask.png"),

            )
            for line in f.readlines()
        ]


def main(
        file_names: Path = Path("data/data_splits/test_200.txt"), 
         output_path: Path = Path("output/")
         ):

    # file_names = Path("data/data_splits/test_200.txt")
    # file_names = Path("data/data_splits/training.txt")    
    # output_path = Path("output/")

    # ensure output path exists
    output_path = output_path / ("inpainted_" + file_names.stem)
    output_path.mkdir(parents=True, exist_ok=True)

    # construct pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    # load images
    file_list = load_file_list(file_names)

    for orig_path, masked_path, mask_path in tqdm(file_list):
        # break
        masked_img = Image.open(str(masked_path))
        mask = Image.open(mask_path)
        # orig_img = Image.open(orig_path)

        prompt = "an image of a cat"
        # gen_image = pipe(prompt=prompt, 
        #                  image=masked_img, 
        #                  mask_image=mask,
        #                  num_inference_steps=1,
        #                  ).images[0]

        # # generate mulitple images
        gen_images1 = pipe(prompt=prompt,
                         image=masked_img,
                         mask_image=mask,
                         num_inference_steps=1,
                         num_images_per_prompt=10,
                         ).images

        gen_images2 = pipe(prompt="",
                         image=masked_img,
                         mask_image=mask,
                         num_inference_steps=1,
                         num_images_per_prompt=10,
                         ).images
    
        gen_images3 = pipe(prompt="the face of a cat",
                         image=masked_img,
                         mask_image=mask,
                         num_inference_steps=1,
                         num_images_per_prompt=10,
                         ).images
        gen_images = gen_images1 + gen_images2 + gen_images3

        # visualize the 10 generated images
        for gen_image in gen_images:
            plt.imshow(gen_image)
            plt.show()
            
        # compute the mean of the generated PIL images
        numpy_images = [numpy.array(gen_image) for gen_image in gen_images]
        mean_image = numpy.mean(numpy_images, axis=0)
        mean_image = Image.fromarray(mean_image.astype(numpy.uint8))
        # plt.imshow(mean_image)
        gen_image = mean_image

        # visualize the generated image
        # plt.imshow(gen_image.cpu().numpy().transpose(1, 2, 0))


        # resize to original size
        gen_image_fix = gen_image.resize(masked_img.size)
        # replace the masked region with generated image
        gen_image_fix = Image.composite(gen_image_fix, masked_img, mask)

        # save the generated image
        gen_image_fix.save(output_path / (orig_path.stem + ".png"))

if __name__ == "__main__":
    main()