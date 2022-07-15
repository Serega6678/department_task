import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from src.metrics import f1, precision, recall

if __name__ == "__main__":
    target_path = Path(sys.argv[1])
    probas_path = target_path.parent / (target_path.stem + "_answer_probas.json")
    preds = []
    targets = []

    with target_path.open() as f:
        data = json.load(f)
        for elem in data:
            targets.extend(elem["labels"])

    with probas_path.open() as f:
        data = json.load(f)
        for elem in data:
            preds.extend(elem["labels"])

    cur_preds = np.array(preds)
    cur_targets = np.array(targets)

    metrics = []
    thrs = np.linspace(0, 1, 101)
    if len(sys.argv) == 3:
        thr = float(sys.argv[2])
        thrs = [thr]
        metrics.append(f1(cur_preds > thr, cur_targets))

    for thr in tqdm(thrs, desc="Finding best threshold"):
        metrics.append(f1(cur_preds > thr, cur_targets))

    best_thr_idx = np.argmax(metrics)
    best_thr = thrs[best_thr_idx]
    f1_metric = metrics[best_thr_idx]
    cur_precision = precision(cur_preds > best_thr, cur_targets)
    cur_recall = recall(cur_preds > best_thr, cur_targets)
    print(
        "Best thr={:.2f}, f1={:.2f}, precision={:.2f}, recall={:.2f}".format(
            float(best_thr),
            float(f1_metric),
            float(cur_precision),
            float(cur_recall)
        )
    )
