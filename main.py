# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import re
from transformers import AutoTokenizer

# ==============================
# إعدادات عامة
# ==============================
MODEL_PATH = "models/intfloat_multilingual-e5-small_merged_int8.onnx"
TOKENIZER_PATH = "models/tool"
TARGET_DIM = 384  # الأبعاد النهائية المطلوبة

# تحميل جلسة ONNX
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
# تحميل التوكنيزر المحلي
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# إنشاء تطبيق FastAPI
app = FastAPI(title="Arabic Text Embedding API")

# ==============================
# دوال مساعدة
# ==============================
def normalize_arabic(text: str) -> str:
    """تنظيف النص العربي من التشكيل والرموز وتوحيد الحروف."""
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text: str):
    """تقسيم النص إلى جمل باستخدام علامات الترقيم."""
    sentences = re.split(r'[.\n:؛؟!]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def reduce_dim(embedding: np.ndarray, target_dim: int = TARGET_DIM) -> np.ndarray:
    """تقليل أبعاد المتجه إلى TARGET_DIM باستخدام إسقاط عشوائي."""
    if embedding.shape[0] == target_dim:
        return embedding
    np.random.seed(42)
    projection = np.random.randn(embedding.shape[0], target_dim).astype(np.float32)
    reduced = embedding @ projection  # ضرب المصفوفة مع 1D vector
    reduced /= np.linalg.norm(reduced) + 1e-10
    return reduced

def embed_text(text: str):
    """تحويل النص إلى قائمة من (جملة، متجه) باستخدام ONNX وتقليل الأبعاد."""
    normalized = normalize_arabic(text)
    sentences = split_sentences(normalized)
    if not sentences:
        return []

    embeddings = []
    for s in sentences:
        input_text = "passage: " + s
        inputs = tokenizer(
            input_text,
            return_tensors="np",
            truncation=True,
            max_length=256
        )
        ort_inputs = {k: v for k, v in inputs.items()}
        ort_outs = session.run(None, ort_inputs)
        vector = ort_outs[0][0]  # قد يكون shape=(seq_len, hidden_size)
        vector = vector.mean(axis=0)  # Pooling عبر المتوسط
        vector = vector / (np.linalg.norm(vector) + 1e-10)  # تطبيع المتجه
        vector = reduce_dim(vector, TARGET_DIM)
        embeddings.append(vector)

    return list(zip(sentences, embeddings))

# ==============================
# نموذج البيانات الوارد
# ==============================
class TextRequest(BaseModel):
    text: str

# ==============================
# نقاط النهاية
# ==============================
@app.get("/")
def root():
    return {"message": "✅ Arabic Text Embedding API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed_endpoint(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ.")
    results = embed_text(request.text)
    if not results:
        raise HTTPException(status_code=400, detail="لم يتم العثور على جمل صالحة في النص.")
    response = [{"sentence": s, "embedding": e.tolist()} for s, e in results]
    return {"results": response}

# ==============================
# تشغيل السيرفر
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
