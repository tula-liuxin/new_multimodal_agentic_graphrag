import os, json
from typing import List
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import get_logger

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_tfidf(cfg, logger):
    data_dir = cfg["data_dir"]
    chunks_path = os.path.join(data_dir, "chunks.jsonl")
    texts = []
    for rec in tqdm(list(read_jsonl(chunks_path)), desc="load chunks", unit="chunk"):
        texts.append(rec["text"] if rec["text"].strip() else " ")
    max_features = int(cfg.get("tfidf_max_features", 200000))
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)
    joblib.dump(vec, os.path.join(data_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(X, os.path.join(data_dir, "tfidf_matrix.joblib"))
    logger.info(f"TF-IDF（词面）完成：{X.shape}")

def run_char_tfidf(cfg, logger):
    data_dir = cfg["data_dir"]
    chunks_path = os.path.join(data_dir, "chunks.jsonl")
    texts = []
    for rec in tqdm(list(read_jsonl(chunks_path)), desc="load chunks", unit="chunk"):
        texts.append(rec["text"] if rec["text"].strip() else " ")
    lo, hi = cfg.get("char_ngram_range", [2,6])
    vec = TfidfVectorizer(analyzer="char", ngram_range=(int(lo), int(hi)))
    X = vec.fit_transform(texts)
    joblib.dump(vec, os.path.join(data_dir, "tfidf_char_vectorizer.joblib"))
    joblib.dump(X, os.path.join(data_dir, "tfidf_char_matrix.joblib"))
    logger.info(f"TF-IDF（字 n-gram）完成：{X.shape}")
