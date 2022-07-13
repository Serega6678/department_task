from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from src.data import AuthorChangeExtractorDataset
from src.model import get_extractor_model_and_tokenizer, get_sentences_embeddings


if __name__ == "__main__":
    model, tokenizer = get_extractor_model_and_tokenizer()

    for dataset_path_input in sys.argv[1:]:
        dataset_path = Path(dataset_path_input)
        dataset_output_folder = dataset_path.parent

        dataset = AuthorChangeExtractorDataset(dataset_path)

        embeddings = []
        for item in tqdm(dataset, desc="Calculating output embeddings for {}".format(dataset_path_input)):
            sentence_embeddings = get_sentences_embeddings(
                item["text"],
                model,
                tokenizer
            )
            embeddings.extend(sentence_embeddings)
        embeddings_np = np.stack(embeddings)

        dataset_path_output = dataset_output_folder / (dataset_path.stem + "_embeddings")
        dataset_path_output_str = str(dataset_path_output)
        np.save(dataset_path_output_str, embeddings_np)
