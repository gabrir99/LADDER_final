import torch
from torch import nn

class Dataset(torch.utils.data.Dataset):
    """
    torch.utils.data.Dataset wrapper converting into tensor
    """

    float_tensors = ["attention_mask"]

    def __init__(self, data: list):
        """
        Takes the dataseat already tokenized and provide the needed apis,
        in order to make the dataloader work with it.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}
