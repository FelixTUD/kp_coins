from dataset import NewCoinDataset, Collator
import torch

d = NewCoinDataset(None)
loader = torch.utils.data.DataLoader(d, batch_size=5 , drop_last=True, shuffle=True, collate_fn=Collator())
n = next(enumerate(loader))