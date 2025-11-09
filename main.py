from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AlbertTokenizer
import numpy as np

app = FastAPI(title="Arabic ALBERT Embedding API")

@app.get("/")
def home():
    return {"message": "Arabic ALBERT Embedding API is running 🚀"}

# ----------------------------
# تحميل tokenizer ONNX
# ----------------------------
TOKENIZER_PATH = "models/asafaya/albert-base-arabic"
MODEL_PATH = "models/albert_arabic_wa_merged.onnx"
TARGET_DIM = 384

try:
    tokenizer = AlbertTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
except Exception as e:
    raise RuntimeError(f"فشل تحميل tokenizer: {e}")

try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
except Exception as e:
    raise RuntimeError(f"فشل تحميل نموذج ONNX: {e}")

# ----------------------------
# Projection matrix لتقليل الأبعاد
# ----------------------------
np.random.seed(42)
projection_matrix = np.random.randn(768, TARGET_DIM).astype(np.float32)

# ----------------------------
# نموذج البيانات
# ----------------------------
class TextInput(BaseModel):
    text: str

# ----------------------------
# Endpoint لتحويل النص إلى embedding
# ----------------------------
@app.post("/embed")
def get_embedding(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ")

    inputs = tokenizer(input.text, return_tensors="np", truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    embedding_768 = outputs[0].mean(axis=1)
    embedding_384 = embedding_768 @ projection_matrix

    return {"embedding": embedding_384[0].tolist()}
