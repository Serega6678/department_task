import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

from src.data import AuthorChangeExtractorDataset
from src.model import get_extractor_model_and_tokenizer, get_sentences_embeddings
from src.pl_module import LightningModule

if __name__ == "__main__":
    json_path = Path(sys.argv[1])
    thr = float(sys.argv[2])
    checkpoint_path = sys.argv[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor_dataset = AuthorChangeExtractorDataset(json_path)

    extractor, tokenizer = get_extractor_model_and_tokenizer()
    classification_model = LightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu"
    ).eval()

    embeddings = []
    ids = []
    for batch in tqdm(extractor_dataset, desc="Calculating embeddings"):
        embeddings.append(torch.from_numpy(get_sentences_embeddings(batch["text"], extractor, tokenizer)))
        ids.append(batch["id"])

    assert len(embeddings) == len(ids)

    result_binary = []
    result_proba = []
    with torch.no_grad():
        for elem, cur_id in tqdm(zip(embeddings, ids), desc="Making predictions"):
            cur_embeddings = elem.unsqueeze(0).to(device)
            predictions = torch.sigmoid(classification_model(cur_embeddings)).cpu().numpy()
            result_binary.append(
                {
                    "id": cur_id,
                    "labels": (predictions > thr).astype(int).tolist()[0]
                }
            )
            result_proba.append(
                {
                    "id": cur_id,
                    "labels": predictions.tolist()[0]
                }
            )

    save_path = json_path.parent / (json_path.stem + "_answer.json")
    with save_path.open("w") as f:
        json.dump(result_binary, f, indent=2)

    save_path = json_path.parent / (json_path.stem + "_answer_probas.json")
    with save_path.open("w") as f:
        json.dump(result_proba, f, indent=2)
