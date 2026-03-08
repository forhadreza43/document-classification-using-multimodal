from dataclasses import dataclass
from typing import Tuple

import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class FaissKNNIndex:
    index: faiss.Index
    train_labels: np.ndarray


def build_faiss_l2_index(train_embeddings: np.ndarray, train_labels: np.ndarray) -> FaissKNNIndex:
    assert train_embeddings.dtype == np.float32
    n, d = train_embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(train_embeddings)
    return FaissKNNIndex(index=index, train_labels=train_labels.astype(np.int64))


def knn1_score_and_neighbor(index: FaissKNNIndex, emb: np.ndarray) -> Tuple[float, int, int]:
    emb2 = emb.reshape(1, -1)
    dist2, idx = index.index.search(emb2, k=1)
    nn_idx = int(idx[0, 0])
    nn_label = int(index.train_labels[nn_idx])
    euclid = float(np.sqrt(dist2[0, 0]))
    score = -euclid
    return score, nn_idx, nn_label


def extract_embeddings_and_logits_multimodal(model, loader: DataLoader, device: str):
    model.eval()
    embs = []
    logs = []
    ys = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract multimodal"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].cpu().numpy().astype(np.int64)

            logits, h = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )

            embs.append(h.cpu().numpy().astype(np.float32))
            logs.append(logits.cpu().numpy().astype(np.float32))
            ys.append(labels)

    embeddings = np.concatenate(embs, axis=0)
    logits = np.concatenate(logs, axis=0)
    labels = np.concatenate(ys, axis=0)
    return embeddings, logits, labels


def estimate_threshold_theta(train_index: FaissKNNIndex, val_embeddings: np.ndarray, tpr_target: float) -> float:
    scores = []
    for i in range(val_embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, val_embeddings[i])
        scores.append(s)

    scores = np.array(scores, dtype=np.float32)
    scores_sorted = np.sort(scores)[::-1]
    n = len(scores_sorted)
    ind = int(np.ceil(tpr_target * n))
    ind0 = max(0, min(n - 1, ind - 1))
    theta = float(scores_sorted[ind0])
    return theta


def knn_predict_no_agreement(train_index: FaissKNNIndex, embeddings: np.ndarray, logits: np.ndarray, theta: float):
    pred = logits.argmax(axis=1).astype(np.int64)
    is_id = np.zeros((embeddings.shape[0],), dtype=bool)
    out_label = np.full((embeddings.shape[0],), -1, dtype=np.int64)

    for i in range(embeddings.shape[0]):
        score, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        if score >= theta:
            is_id[i] = True
            out_label[i] = pred[i]
        else:
            is_id[i] = False
            out_label[i] = -1

    return is_id, out_label


def knn_star_predict(train_index: FaissKNNIndex, embeddings: np.ndarray, logits: np.ndarray, theta: float):
    pred = logits.argmax(axis=1).astype(np.int64)
    is_id = np.zeros((embeddings.shape[0],), dtype=bool)
    out_label = np.full((embeddings.shape[0],), -1, dtype=np.int64)

    for i in range(embeddings.shape[0]):
        score, _, nn_label = knn1_score_and_neighbor(train_index, embeddings[i])

        if pred[i] != nn_label:
            is_id[i] = False
            out_label[i] = -1
            continue

        if score >= theta:
            is_id[i] = True
            out_label[i] = pred[i]
        else:
            is_id[i] = False
            out_label[i] = -1

    return is_id, out_label