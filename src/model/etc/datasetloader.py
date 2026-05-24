import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class DictionaryDataset(Dataset):
    def __init__(self, data_list, labels, stoi_map, max_len=512):
        self.data_list = data_list
        self.labels = labels
        self.stoi = stoi_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def flatten(self, data):
        """Recursively flattens any dict/list into a string."""
        if isinstance(data, dict):
            return " ".join([f"{k}:{self.flatten(v)}" for k, v in data.items()])
        elif isinstance(data, list):
            return ",".join([self.flatten(i) for i in data])
        return str(data)

    def __getitem__(self, idx):
        # 1. Flatten the dictionary to string
        raw_string = self.flatten(self.data_list[idx])
        
        # 2. Encode to integers
        encoded = [self.stoi.get(c, 0) for c in raw_string]
        
        # 3. Truncate if too long
        encoded = encoded[:self.max_len]
        
        return torch.tensor(encoded), torch.tensor(self.labels[idx])

def collate_fn(batch):
    """Pads sequences in a batch to the same length."""
    data, labels = zip(*batch)
    # Pad with 0 (usually the index for 'unknown' or 'padding')
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    return data_padded, torch.stack(labels)
