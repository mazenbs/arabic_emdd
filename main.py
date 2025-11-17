import re
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

# ==============================
# إعداد المسارات بشكل آمن لـ Render
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intfloat_multilingual-e5-small_merged_int8.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tool")

# تحقق من وجود الملفات قبل التشغيل
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ ملف النموذج غير موجود: {MODEL_PATH}")
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"❌ مجلد التوكنيزر غير موجود: {TOKENIZER_PATH}")

# تحميل النموذج والتوكنيزر
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

app = FastAPI(title="Arabic Search Embedding API")

# ==============================
# دوال مساعدة
# ==============================
def normalize_arabic(text: str) -> str:
    """تطبيع النص العربي."""
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة\b", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def embed_search_text(text: str):
    """
    تحويل نص البحث إلى متجه 384-D ثابت.
    ✅ يستخدم التطبيع فقط، بدون تقسيم الجمل
    """
    if not text.strip():
        return None

    # تطبيع النص
    text = normalize_arabic(text)

    # تجهيز الإدخال للنموذج
    input_text = "passage: " + text
    inputs = tokenizer(
        input_text,
        return_tensors="np",
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outs = session.run(None, ort_inputs)

    vec_seq = ort_outs[0]  # الشكل: (seq_len, 384)
    vec = np.mean(vec_seq, axis=0)  # متوسط → 384-D
    vec = vec / (np.linalg.norm(vec) + 1e-10)  # تطبيع

    return vec.astype(np.float32)

# ==============================
# نموذج الإدخال
# ==============================
class TextRequest(BaseModel):
    text: str

# ==============================
# المسارات
# ==============================
@app.get("/")
def root():
    return {"message": "✅ Arabic Search Embedding API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed_endpoint(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ.")
    
    embedding = embed_search_text(request.text)
    if embedding is None:
        raise HTTPException(status_code=400, detail="لم يتم العثور على نص صالح.")
    
    return {"embedding": embedding.tolist()}

# ==============================
# تشغيل محلي / Render
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)