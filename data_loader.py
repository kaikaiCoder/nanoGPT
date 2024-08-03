import os
import numpy as np
import torch


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:

    def __init__(self, B, T, process_rank, num_process, split):
        self.B = B
        self.T = T
        self.procee_rank = process_rank
        self.num_process = num_process
        assert split in {"train", "val"}

        data_dir = "edu_fineweb10B"
        shard = os.listdir(data_dir)
        shard = [s for s in shard if split in s]
        shard = sorted(shard)
        shard = [os.path.join(data_dir, s) for s in shard]
        self.shard = shard
        assert len(shard) > 0, f"no data files found in {data_dir}"
        if process_rank == 0:
            print(f"found {len(shard)} shards for {split}")
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.procee_rank
        self.current_shard = 0
        self.tokens = load_tokens(self.shard[self.current_shard])

    def next_batch(self):
        B, T, start = self.B, self.T, self.current_position
        end = self.current_position + B * T + 1
        buf = self.tokens[start:end]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_process
        if self.current_position + B * T * self.num_process + 1 > len(self.tokens):
            # if we reach the end of the shard, move to the next shard
            self.current_shard = (self.current_shard + 1) % len(self.shard)
            self.tokens = load_tokens(self.shard[self.current_shard])
            self.current_position = B * T * self.procee_rank
        return x, y
