import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, encodings, train=True):
        self.encodings = encodings
        self.keys = ['input_ids', 'attention_mask']
        if train:
            self.keys += ['start_positions', 'end_positions', 'domain']
        assert(all(key in self.encodings for key in self.keys))

    def __getitem__(self, idx):
        return {key : torch.tensor(self.encodings[key][idx]) for key in self.keys}

    def __len__(self):
        return len(self.encodings['input_ids'])