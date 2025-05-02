from kedro.io import AbstractDataset
import torch
from typing import Any

class TorchTokenizedDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def load(self) -> Any:
        return torch.load(self._filepath)
    
    def save(self, data) -> None:
        torch.save(data, self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath, class_name=self.__class__)