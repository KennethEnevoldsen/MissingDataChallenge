import argparse
import multiprocessing
import os
import pathlib
from functools import partial

from skimage import io
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from tqdm import tqdm

from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list


def compute_inpaint_metrics(org_img, inpainted_img):
    mse_val = mean_squared_error(org_img, inpainted_img)
    ssim = structural_similarity(org_img, inpainted_img, channel_axis=2)
    psnr = peak_signal_noise_ratio(org_img, inpainted_img)

    return {"mse": mse_val, "ssim": ssim, "psnr": psnr}


def compute_metrics(idx, input_data_dir, inpainted_result_dir):
    org_image_name = os.path.join(input_data_dir, "originals", f"{idx}.jpg")
    inpainted_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

    im_org = io.imread(org_image_name)
    im_inpainted = io.imread(inpainted_image_name)

    metrics = compute_inpaint_metrics(im_org, im_inpainted)
    return f'{idx}, {metrics["mse"]}, {metrics["ssim"]}, {metrics["psnr"]}\n'


def evaluate_inpainting(settings):
    input_data_dir = settings["dirs"]["input_data_dir"]
    output_data_dir = settings["dirs"]["output_data_dir"]
    data_set = settings["data_set"]
    inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")

    result_dir = os.path.join(output_data_dir, "evaluations")
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    evaluation_file = os.path.join(result_dir, f"{data_set}_results.csv")

    print(f"Evaluating {data_set} and placing evaluations in {evaluation_file}")

    file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
    file_ids = read_file_list(file_list)
    if file_ids is None:
        return

    f = open(evaluation_file, "w")
    print(f"Evaluating {len(file_ids)} images")

    mse, ssim, psnr = [], [], []
    f.write("id, mse, ssim, psnr\n")
    # f.write("id,mse,ssim,psnr\n")

    _compute_metrics = partial(
        compute_metrics,
        input_data_dir=input_data_dir,
        inpainted_result_dir=inpainted_result_dir,
    )
    with multiprocessing.Pool(4) as pool:
        output = pool.imap(_compute_metrics, tqdm(file_ids))
        for line in output:
            f.write(line)
            idx, mse_val, ssim_val, psnr_val = line.split(",")
            mse.append(float(mse_val))
            ssim.append(float(ssim_val))
            psnr.append(float(psnr_val))

    f.close()

    print(f"Average MSE: {sum(mse) / len(mse):.2f}")
    print(f"Average SSIM: {sum(ssim) / len(ssim):.2f}")
    print(f"Average PSNR: {sum(psnr) / len(psnr):.2f}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="EvaluateInPaintings")
    config = InPaintConfig(args)
    if config.settings is not None:
        evaluate_inpainting(config.settings)
