from typing import Any
from kedro.io import AbstractDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os


class HuggingFaceTokenizer(AbstractDataset):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._handle_login()

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

    def _handle_login(self) -> None:
        try:
            load_dotenv()
            login(token=os.getenv("hf_token"))
        except ValueError as e:
            raise Exception(f"HF key error: {e}")


class HuggingFaceCausalModel(AbstractDataset):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._handle_login()

    def load(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model

    def save(self, data) -> None:
        raise NotImplementedError(
            f"Saving not supported yet for the class {self.__class__}."
        )

    def _handle_login(self) -> None:
        try:
            load_dotenv()
            login(token=os.getenv("hf_token"))
        except ValueError as e:
            raise Exception(f"HF key error: {e}")
