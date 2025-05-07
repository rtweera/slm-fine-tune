from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class TransformerInput:
    # Required for all models
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None  # Often used in BERT for sentence pairs

    # Optional for classification/regression tasks
    labels: Optional[Union[int, float, List[int]]] = None

    # For token classification (NER, POS tagging)
    token_labels: Optional[List[int]] = None

    # For QA (extractive)
    start_positions: Optional[int] = None
    end_positions: Optional[int] = None

    # For sequence-to-sequence tasks (e.g., translation, summarization)
    decoder_input_ids: Optional[List[int]] = None
    decoder_attention_mask: Optional[List[int]] = None

    # For multimodal inputs (e.g., vision-language models)
    pixel_values: Optional[List[float]] = None  # Or np.ndarray / torch.Tensor

    # For multiple choice tasks
    choice_input_ids: Optional[List[List[int]]] = None
    choice_attention_mask: Optional[List[List[int]]] = None
