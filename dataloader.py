from pytorch_lightning import LightningDataModule
from torchvision import transforms


class DepthDataModule(LightningDataModule):
    def prepare_data(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
    
    def test_dataloader(self):
        pass
