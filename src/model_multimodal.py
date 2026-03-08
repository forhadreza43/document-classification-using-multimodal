import torch
import torch.nn as nn
from transformers import LayoutLMv3Model


class LayoutLMv3DocClassifier(nn.Module):
    """
    Outputs:
      - logits
      - CLS embedding from last hidden layer
    """
    def __init__(self, model_name: str, num_classes: int = 16):
        super().__init__()
        self.encoder = LayoutLMv3Model.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
        bbox,
        pixel_values,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state
        h = last_hidden[:, 0, :]   # CLS token
        logits = self.classifier(h)
        return logits, h