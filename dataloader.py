from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from dataset import DepthDataset


class DepthDataModule(LightningDataModule):
    def __init__(self, rgb_dir: str, depth_dir: str, batch_size: int = 8, train_val_test_ratio = [0.7, 0.1, 0.2]):
        super().__init__()
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.batch_size = batch_size
        self.train_val_test_ratio = train_val_test_ratio

    def prepare_data(self):
        depth_dataset = DepthDataset(self.rgb_dir, self.depth_dir)
        n_train_samples = len(depth_dataset) * (self.train_val_test_ratio[0] / sum(self.train_val_test_ratio))
        n_val_samples = len(depth_dataset) * (self.train_val_test_ratio[1] / sum(self.train_val_test_ratio))
        n_test_samples = len(depth_dataset) - (n_train_samples + n_val_samples)
        self.train_dataset, self.val_dataset, self.test_dataset =  random_split(depth_dataset, [n_train_samples, n_val_samples, n_test_samples])

    def train_dataloader(self):
        DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        DataLoader(self.test_dataset, batch_size=self.batch_size)
