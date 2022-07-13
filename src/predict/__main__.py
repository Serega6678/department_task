import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data import AuthorChangeExtractorDataset
from src.model import get_extractor_model_and_tokenizer, get_sentences_embeddings
from src.pl_module import LightningModule

if __name__ == "__main__":
    json_path = Path(sys.argv[1])
    thr = float(sys.argv[2])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor_dataset = AuthorChangeExtractorDataset(json_path)

    extractor, tokenizer = get_extractor_model_and_tokenizer()
    classification_model = LightningModule.load_from_checkpoint(
        "data/checkpoints/last.ckpt",
        map_location="cpu"
    )

    embeddings = []
    ids = []
    for batch in extractor_dataset:
        embeddings.append(torch.from_numpy(get_sentences_embeddings(batch["text"], extractor, tokenizer)))
        ids.append(batch["id"])

    assert len(embeddings) == len(ids)

    result = []
    with torch.no_grad():
        for elem, cur_id in tqdm(zip(embeddings, ids)):
            cur_embeddings = elem.unsqueeze(0).to(device)
            predictions = torch.sigmoid(classification_model(cur_embeddings)).cpu().numpy()
            result.append(
                {
                    "id": cur_id,
                    "labels": (predictions > thr).astype(int).tolist()
                }
            )

    save_path = json_path.parent / (json_path.stem + "_answer.json")
    with save_path.open("w") as f:
        json.dump(result, f, indent=2)
