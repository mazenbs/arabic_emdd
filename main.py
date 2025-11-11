from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AlbertTokenizer
import numpy as np
import re

app = FastAPI(title="Arabic ALBERT Embedding API")

# ==============================
# تفعيل CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# الإعدادات العامة
# ==============================
TOKENIZER_PATH = "models/asafaya/albert-base-arabic"
MODEL_PATH = "models/albert_arabic_wa_merged.onnx"
TARGET_DIM = 384
CHUNK_SIZE = 50
SENTENCE_SPLIT_REGEX = r'(?<=[.!؟])\s+'

# ==============================
# تحميل النموذج
# ==============================
tokenizer = AlbertTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ==============================
# مصفوفة إسقاط محسّنة
# ==============================
np.random.seed(42)
projection_matrix = np.random.normal(0, 0.1, (768, TARGET_DIM)).astype(np.float32)

# ==============================
# نموذج البيانات
# ==============================
class TextInput(BaseModel):
    text: str
    normalize: bool = True
    return_dim: int = TARGET_DIM
    mean_pooling: bool = True

# ==============================
# تقسيم النص إلى جمل
# ==============================
def split_text_to_sentences(text):
    sentences = re.split(SENTENCE_SPLIT_REGEX, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# ==============================
# تقسيم الجملة إلى chunks صغيرة
# ==============================
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# ==============================
# حساب embedding لكل chunk
# ==============================
def compute_embedding(text, mean_pooling=True, normalize=True, return_dim=TARGET_DIM):
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
    outputs = session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
    last_hidden = outputs[0]

    if mean_pooling:
        embedding = last_hidden.mean(axis=1)
    else:
        embedding = last_hidden[:, 0, :]

    embedding_projected = embedding @ projection_matrix[:, :return_dim]

    if normalize:
        norm = np.linalg.norm(embedding_projected, axis=1, keepdims=True)
        embedding_projected = embedding_projected / (norm + 1e-10)

    return embedding_projected[0]

# ==============================
# تحويل النص إلى embedding مع تقسيم الجمل و chunks
# ==============================
def text_to_embedding(text, mean_pooling=True, normalize=True, return_dim=TARGET_DIM):
    if not text.strip():
        raise ValueError("النص فارغ")

    sentences = split_text_to_sentences(text)
    all_chunk_embeddings = []

    for sentence in sentences:
        chunks = chunk_text(sentence, CHUNK_SIZE)
        chunk_embeddings = [compute_embedding(chunk, mean_pooling, normalize, return_dim) for chunk in chunks]
        all_chunk_embeddings.extend(chunk_embeddings)

    final_embedding = np.mean(np.stack(all_chunk_embeddings), axis=0)
    if normalize:
        final_embedding /= (np.linalg.norm(final_embedding) + 1e-10)

    return final_embedding, sentences

# ==============================
# POST /embed
# ==============================
@app.post("/embed")
def embed_text(data: TextInput):
    try:
        final_embedding, sentences = text_to_embedding(
            data.text, data.mean_pooling, data.normalize, data.return_dim
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "num_chunks": len(sentences),
        "chunks": sentences,
        "embedding": final_embedding.tolist()
    }

# ==============================
# GET /embed
# ==============================
@app.get("/embed")
def embed_text_get(
    text: str = Query(..., description="النص المطلوب تحويله إلى متجه"),
    normalize: bool = Query(True),
    mean_pooling: bool = Query(True),
    return_dim: int = Query(TARGET_DIM),
):
    try:
        final_embedding, sentences = text_to_embedding(
            text, mean_pooling, normalize, return_dim
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "num_chunks": len(sentences),
        "chunks": sentences,
        "embedding": final_embedding.tolist()
    }

# ==============================
# Health check
# ==============================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.split("/")[-1],
        "tokenizer": TOKENIZER_PATH.split("/")[-1],
    }

@app.get("/")
def home():
    return {"message": "Arabic ALBERT Embedding API is running 🚀"}
