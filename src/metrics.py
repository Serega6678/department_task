import numpy as np


def precision(preds: np.ndarray, targets: np.ndarray) -> float:
    cor = (preds * targets).sum()
    par = (targets[:-1] * preds[1:] + targets[1:] * preds[:-1]).sum()
    act = targets.sum()
    if act == 0:
        return 0
    return (cor + par / 2) / act


def recall(preds: np.ndarray, targets: np.ndarray) -> float:
    cor = (preds * targets).sum()
    par = (targets[:-1] * preds[1:] + targets[1:] * preds[:-1]).sum()
    pos = preds.sum()
    if pos == 0:
        return 0
    return (cor + par / 2) / pos


def f1(preds: np.ndarray, targets: np.ndarray) -> float:
    cur_precision = precision(preds, targets)
    cur_recall = recall(preds, targets)
    if cur_precision + cur_recall == 0:
        return 0
    cur_f1 = 2 * cur_precision * cur_recall / (cur_precision + cur_recall)
    return cur_f1
