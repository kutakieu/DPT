from argparse import ArgumentParser
from ast import arg
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


def read_args():
    parser = ArgumentParser()
    parser.add_argument('path', type=str, default="data/StrayScanner", help="Path to StrayScanner dataset folders to process.")
    parser.add_argument('--output_depth_dir', type=str, default="data/processed/depth", help="Path directory to save depth images.")
    parser.add_argument('--output_rgb_dir', type=str, default="data/processed/rgb", help="Path directory to save rgb images.")
    parser.add_argument('--save_frame_interval', '-f', default=3, help="frame intervals to save as dataset")
    parser.add_argument('--confidence', '-c', type=int, default=None,
            help="Keep only depth estimates with confidence equal or higher to the given value. There are three different levels: 0, 1 and 2. Higher is more confident.")
    return parser.parse_args()



def process_one_stray_scanner_folder(input_dir, output_depth_dir, output_rgb_dir, save_frame_interval=1):
    rgb_video_frames = extract_frames_from_video(str(input_dir / "rgb.mp4"))

    depth_dir = input_dir / "depth"
    depth_img_paths = list(depth_dir.glob("*.png"))
    depth_img_paths.sort(key=lambda x: int(x.stem))
    # TODO: deal with the confidence data
    # confidence_dir = input_dir / "confidence"
    # confidence_img_paths = list(confidence_dir.glob("*.png")).sort(key=lambda x: int(x.stem))

    for i in range(0, min(len(rgb_video_frames), len(depth_img_paths)), save_frame_interval):
        rgb_video_frame = rgb_video_frames[i]
        depth_img_path = depth_img_paths[i]
        frame_id = uuid4()

        rgb_image = Image.fromarray(rgb_video_frame)
        depth_img = Image.open(depth_img_path)

        rgb_image.save(output_rgb_dir / f"{frame_id}.jpg")
        depth_img.save(output_depth_dir / f"{frame_id}.png")


def extract_frames_from_video(video_filepath):
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        return

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"processing {video_filepath}")
    print(f"# of video frames: {n_frames}")

    video_fremes = []
    while True:
        ret, frame = cap.read()
        if ret:
            video_fremes.append(frame)
        else:
            return video_fremes

def main(args):
    output_depth_dir = Path(args.output_depth_dir)
    if not output_depth_dir.exists():
        output_depth_dir.mkdir(parents=True)
    
    output_rgb_dir = Path(args.output_rgb_dir)
    if not output_rgb_dir.exists():
        output_rgb_dir.mkdir(parents=True)

    data_dirs = [path for path in Path(args.path).glob("*") if path.is_dir()]
    for data_dir in data_dirs:
        process_one_stray_scanner_folder(data_dir, output_depth_dir, output_rgb_dir, args.save_frame_interval)

if __name__ == "__main__":
    main(read_args())
