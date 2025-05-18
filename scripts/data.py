import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

class DakshinaDataset(Dataset):
    def __init__(self, file_path, input_char2idx=None, target_char2idx=None, build_vocab=False):
        self.pairs = []
        self.input_vocab = set()
        self.target_vocab = set()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    fields = line.strip().split('\t')
                    if len(fields) >= 2:
                        target, source = fields[0], fields[1]
                        self.pairs.append((source, target))
                        if build_vocab:
                            self.input_vocab.update(source)
                            self.target_vocab.update(target)


        if build_vocab:
            self.input_vocab = sorted(self.input_vocab)
            self.target_vocab = sorted(self.target_vocab)

            self.input_char2idx = {c: i + 1 for i, c in enumerate(self.input_vocab)}
            self.input_char2idx['<pad>'] = 0

            self.target_char2idx = {c: i + 3 for i, c in enumerate(self.target_vocab)}
            self.target_char2idx['<pad>'] = 0
            self.target_char2idx['<sos>'] = 1
            self.target_char2idx['<eos>'] = 2

            self.input_idx2char = {i: c for c, i in self.input_char2idx.items()}
            self.target_idx2char = {i: c for c, i in self.target_char2idx.items()}
        else:
            self.input_char2idx = input_char2idx
            self.target_char2idx = target_char2idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        x_seq = [self.input_char2idx[c] for c in src]
        y_seq = [self.target_char2idx['<sos>']] + [self.target_char2idx[c] for c in tgt] + [self.target_char2idx['<eos>']]
        return torch.tensor(x_seq), torch.tensor(y_seq)

def collate_fn(batch):
    x_seqs, y_seqs = zip(*batch)
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=0)
    return x_padded, y_padded

def get_data_loaders(base_path, batch_size=32):
    """
    Prepares train/dev/test dataloaders from Dakshina dataset at the specified base_path.
    Returns: train_loader, dev_loader, test_loader, input_char2idx, target_char2idx
    """
    train_path = os.path.join(base_path, "hi.translit.sampled.train.tsv")
    dev_path   = os.path.join(base_path, "hi.translit.sampled.dev.tsv")
    test_path  = os.path.join(base_path, "hi.translit.sampled.test.tsv")

    # Load and build vocab
    train_dataset = DakshinaDataset(train_path, build_vocab=True)

    # Use same vocab for dev/test
    dev_dataset = DakshinaDataset(dev_path,
                                  input_char2idx=train_dataset.input_char2idx,
                                  target_char2idx=train_dataset.target_char2idx)

    test_dataset = DakshinaDataset(test_path,
                                   input_char2idx=train_dataset.input_char2idx,
                                   target_char2idx=train_dataset.target_char2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, train_dataset.input_char2idx, train_dataset.target_char2idx

base_path = "./data/dakshina_dataset_v1.0/hi/lexicons"
train_loader, dev_loader, test_loader, input_char2idx, target_char2idx = get_data_loaders(base_path)

# Check one batch
for x, y in train_loader:
    print(x.shape, y.shape)  # e.g. torch.Size([32, max_len])
    break
