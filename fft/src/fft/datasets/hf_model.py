from typing import Any
from kedro.io import AbstractDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

class HuggingFaceTokenizer(AbstractDataset):
    def __init__(self, model_name: str, credentials: dict = None):
        self.model_name = model_name
        self._handle_login(credentials)

    def load(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def save(self, data: AutoTokenizer) -> None:
        raise NotImplementedError(
            f"Saving not supported yet for the class {self.__class__}."
        )

    def _describe(self) -> dict[str, Any]:
        return dict(
            model_name=self.model_name, data_type="tokenizer", class_name=self.__class__
        )

    def _handle_login(self, credentials: dict) -> None:
        token = credentials['token'] if credentials else None
        try:
            login(token=token)
        except ValueError as e:
            raise Exception(f"HF key error: {e}")


class HuggingFaceCausalModel(AbstractDataset):
    def __init__(self, model_name: str, credentials: dict = None):
        self.model_name = model_name
        self._handle_login(credentials)

    def load(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model

    def save(self, data) -> None:
        raise NotImplementedError(
            f"Saving not supported yet for the class {self.__class__}."
        )

    def _handle_login(self, credentials: dict) -> None:
        token = credentials['token'] if credentials else None
        try:
            login(token=token)
        except ValueError as e:
            raise Exception(f"HF key error: {e}")

    def _describe(self) -> dict[str, Any]:
        return dict(
            model_name=self.model_name, data_type="model", class_name=self.__class__
        )