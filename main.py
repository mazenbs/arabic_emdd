from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import AlbertTokenizer
import threading

# =========================================================
# إعداد التطبيق
# =========================================================
app = FastAPI(title="Improved Arabic ALBERT Embedding API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# متغيرات عامة (lazy loading)
# =========================================================
MODEL_PATH = "models/albert_arabic_wa_merged.onnx"
TOKENIZER_PATH = "models/asafaya/albert-base-arabic"

TARGET_DIM_DEFAULT = 384
embedding_dim_original = 768

tokenizer = None
session = None
projection_matrix = None
model_lock = threading.Lock()

# =========================================================
# تحميل النموذج عند الطلب فقط (Lazy Load)
# =========================================================
def load_model():
    global tokenizer, session, projection_matrix

    with model_lock:
        if tokenizer is not None and session is not None:
            return

        print("📦 تحميل tokenizer والنموذج ...")
        tokenizer = AlbertTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )

        print("📊 تحميل مصفوفة الإسقاط (PCA) محسّنة ...")
        # محاكاة مصفوفة PCA مدربة مسبقًا (في مشروع حقيقي يجب حسابها من بيانات حقيقية)
        np.random.seed(42)
        pca_matrix = np.random.randn(embedding_dim_original, TARGET_DIM_DEFAULT).astype(np.float32)
        # تطبيع الأعمدة لتحسين الثبات العددي
        pca_matrix /= np.linalg.norm(pca_matrix, axis=0, keepdims=True)
        projection_matrix = pca_matrix

        print("✅ النموذج جاهز للاستخدام.")

# =========================================================
# نموذج الإدخال
# =========================================================
class TextInput(BaseModel):
    text: str
    normalize: bool = True
    reduce_dim: int = TARGET_DIM_DEFAULT
    pooling: str = "mean"  # "mean" أو "cls"

# =========================================================
# Health Check
# =========================================================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": session is not None}

# =========================================================
# الصفحة الرئيسية
# =========================================================
@app.get("/")
def home():
    return {"message": "Improved Arabic ALBERT Embedding API is running 🚀"}

# =========================================================
# تحويل النص إلى embedding
# =========================================================
@app.post("/embed")
def get_embedding(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ")

    if tokenizer is None or session is None:
        load_model()

    # ترميز النص
    inputs = tokenizer(
        input.text,
        return_tensors="np",
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # تمرير البيانات للنموذج
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    last_hidden_state = outputs[0]

    # اختيار طريقة pooling
    if input.pooling == "cls":
        embedding_768 = last_hidden_state[:, 0, :]  # أول توكن
    else:
        # mean pooling مع مراعاة attention mask
        mask = attention_mask[..., None]
        sum_embeddings = np.sum(last_hidden_state * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embedding_768 = sum_embeddings / sum_mask

    # إسقاط الأبعاد إلى الهدف
    reduce_dim = min(input.reduce_dim, projection_matrix.shape[1])
    projection_sub = projection_matrix[:, :reduce_dim]
    embedding_reduced = embedding_768 @ projection_sub

    # تطبيع إذا طُلب
    if input.normalize:
        norms = np.linalg.norm(embedding_reduced, axis=1, keepdims=True)
        embedding_reduced = embedding_reduced / np.clip(norms, 1e-9, None)

    return {
        "embedding": embedding_reduced[0].tolist(),
        "shape": list(embedding_reduced.shape),
        "options": {
            "normalize": input.normalize,
            "reduce_dim": reduce_dim,
            "pooling": input.pooling,
        },
    }