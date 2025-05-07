# This is a monkey-patch for HFDataset to avoid versioning issues
# when using the Hugging Face datasets library with Kedro.

from kedro_datasets.huggingface import HFDataset


class PatchedHFDataset(HFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._version = None  # Dummy attribute to satisfy versioning check
