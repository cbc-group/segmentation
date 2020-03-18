"""
Process the entire dataset.
"""
from pytorch3dunet.unet3d.utils import get_slice_builder, ConfigDataset


class TIFFDataset(ConfigDataset):
    def __init__(self, root, phase):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


    @classmethod
    def create_datasets(cls, dataset_config, phase):
        pass