from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from emdd import text_to_vector, chunk_text, TARGET_DIM  # ← نستخدم الملف emdd.py

# ==============================
# إنشاء تطبيق FastAPI
# ==============================
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
# نموذج البيانات المستقبلة
# ==============================
class TextInput(BaseModel):
    text: str
    normalize: bool = True
    return_dim: int = TARGET_DIM
    mean_pooling: bool = True

# ==============================
# POST /embed
# ==============================
@app.post("/embed")
def embed_text(data: TextInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ")

    try:
        vector = text_to_vector(
            data.text,
            mean_pooling=data.mean_pooling,
            normalize=data.normalize,
            return_dim=data.return_dim,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "num_chunks": len(chunk_text(data.text)),
        "embedding": vector.tolist(),
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
    if not text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ")

    try:
        vector = text_to_vector(
            text,
            mean_pooling=mean_pooling,
            normalize=normalize,
            return_dim=return_dim,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "num_chunks": len(chunk_text(text)),
        "embedding": vector.tolist(),
    }

# ==============================
# Health check
# ==============================
@app.get("/health")
def health():
    return {"status": "ok", "source": "emdd.py"}

# ==============================
# الصفحة الرئيسية
# ==============================
@app.get("/")
def home():
    return {"message": "Arabic ALBERT Embedding API (using emdd.py) 🚀"}
