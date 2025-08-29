import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens)-max_len, stride):
            input_chunk = tokens[i:i+max_len]
            target_chunk = tokens[i+1:i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids) 

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, max_len=256, 
                         stride=128, 
                         batch_size=4, 
                         shuffle=True, 
                         drop_last=True, 
                         num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDatasetV1(text, tokenizer, max_len, stride)

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )

    return dataloader