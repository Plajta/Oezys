#
# All the loading utilities + Dataset + Dataloader classes
#

import yaml
from torch.utils.data import Dataset, DataLoader


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config


class TearDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem_(self, idx):
        pass

class TearDataloader:
    def __init__(self, data):
        # TODO: add batch-sizes etc.
        self.dataloader = DataLoader(data, )