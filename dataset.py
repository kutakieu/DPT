from pathlib import PosixPath

from torch.utils.data import Dataset
from torchvision.io import read_image


class DepthDataset(Dataset):
    def __init__(self, rgb_dir: PosixPath, depth_dir: PosixPath, transform=None, target_transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.target_transform = target_transform
        self.rgb_paths = [file for file in self.rgb_dir.glob("*") if not str(file.name).startswith(".")]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_dir / f"{self.rgb_paths[idx].stem}.png"
        rgb_img = read_image(rgb_path)
        depth_img = read_image(depth_path)
        rgb_img = self.transform(rgb_img)
        depth_img = self.target_transform(depth_img)
        return rgb_img, depth_img
