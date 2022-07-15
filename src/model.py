import typing as tp

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def get_extractor_model_and_tokenizer(
        name: str = "cointegrated/rubert-tiny",
        device: str = "cpu"
) -> tp.Tuple[tp.Callable, tp.Callable]:
    model = AutoModel.from_pretrained(name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def get_sentences_embeddings(text: tp.List[str], model: tp.Callable, tokenizer: tp.Callable) -> np.ndarray:
    device = model.device if hasattr(model, "device") else "cpu"
    text = list(map(lambda x: x.lower(), text))
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()


class AuthorClassifier(nn.Module):
    def __init__(
        self,
        emb_dim: int = 312,
        extractor_hidden: int = 256,
        classifier_hidden: int = 128,
        extractor_num_layers: int = 4
    ) -> None:
        super().__init__()
        self.extractor = nn.GRU(
            input_size=emb_dim,
            hidden_size=extractor_hidden,
            num_layers=extractor_num_layers,
            batch_first=True,
            bidirectional=False,
            bias=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(extractor_hidden, classifier_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(classifier_hidden, 1, bias=False)
        )

    def forward(self, x) -> torch.Tensor:
        features, _ = self.extractor(x)
        return self.classifier(features).squeeze(-1)
