import json
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AuthorChangeExtractorDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()
        with path.open() as f:
            self.data: tp.List[tp.Dict[str, tp.Union[tp.List[tp.List[str]], tp.List[int]]]] = json.load(f)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[str, tp.Union[tp.List[str], tp.List[int]]]]:
        ans = {}
        cur_data = self.data[idx]
        ans["id"] = cur_data["id"]
        ans["text"] = list(map(" ".join, cur_data["text"]))
        labels = cur_data.get("labels")
        if labels is not None:
            ans["labels"] = labels
        return ans

    def __len__(self) -> int:
        return len(self.data)


class AuthorChangeFromNumpyDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.embeddings = np.load(str(path))

        initial_data_path = path.parent / (path.stem.replace("_embeddings", "") + ".json")
        with initial_data_path.open() as f:
            self.json_data: tp.List[tp.Dict[str, tp.Union[tp.List[tp.List[str]], tp.List[int]]]] = json.load(f)
        initial_offset = 0
        self.idx_to_offset = []
        for cur_data in self.json_data:
            self.idx_to_offset.append(initial_offset)
            initial_offset += len(cur_data["labels"])
        self.idx_to_offset.append(initial_offset)

    def __getitem__(self, idx: int) -> tp.Dict[str, np.ndarray]:
        embeddings = self.embeddings[self.idx_to_offset[idx]: self.idx_to_offset[idx + 1]]
        return {
            "embeddings": embeddings,
            "labels": np.array(self.json_data[idx]["labels"])
        }

    def __len__(self):
        return len(self.json_data)


def collate_fn(batch: tp.List[tp.Dict[str, np.ndarray]]) -> tp.Dict[str, torch.Tensor]:
    embeddings = []
    labels = []
    for item in batch:
        embeddings.append(torch.from_numpy(item["embeddings"]))
        labels.append(torch.from_numpy(item["labels"]))
    return {
        "embeddings": pad_sequence(embeddings, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-1)
    }
