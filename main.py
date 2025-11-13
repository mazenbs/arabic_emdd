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
TARGET_DIM = 384  # عدد الأبعاد النهائي المطلوب

# ==============================
# إعداد ONNX مع تحسين الأداء
# ==============================
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# تحميل الموديل والجلسة لمرة واحدة فقط
session = ort.InferenceSession(MODEL_PATH, sess_options=options, providers=['CPUExecutionProvider'])

# تحميل التوكنيزر المحلي فقط
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# إنشاء تطبيق FastAPI
app = FastAPI(title="Arabic Text Embedding API", version="1.0")

# ==============================
# دوال مساعدة
# ==============================
def normalize_arabic(text: str) -> str:
    """تنظيف النص العربي من الرموز والتشكيل وتوحيد الحروف."""
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def reduce_dim(embedding: np.ndarray, target_dim: int = TARGET_DIM) -> np.ndarray:
    """تقليل الأبعاد إلى TARGET_DIM بإسقاط عشوائي ثابت."""
    if embedding.shape[0] == target_dim:
        return embedding
    np.random.seed(42)
    projection = np.random.randn(embedding.shape[0], target_dim).astype(np.float32)
    reduced = embedding @ projection
    reduced /= np.linalg.norm(reduced) + 1e-10
    return reduced

def embed_text(text: str):
    """تحويل النص إلى متجه واحد (embedding)."""
    normalized = normalize_arabic(text)
    if not normalized:
        return None

    # نستخدم النص كاملاً بدل تقسيمه لجمل لتسريع العملية
    input_text = "passage: " + normalized
    inputs = tokenizer(
        input_text,
        return_tensors="np",
        truncation=True,
        max_length=256
    )
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outs = session.run(None, ort_inputs)
    vector = ort_outs[0][0]
    vector = vector / (np.linalg.norm(vector) + 1e-10)
    return reduce_dim(vector).astype(np.float32)

# ==============================
# نموذج الإدخال
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
async def embed_endpoint(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ.")
    embedding = embed_text(request.text)
    if embedding is None:
        raise HTTPException(status_code=400, detail="لم يتم إنشاء متجه صالح.")
    return {"embedding": embedding.tolist()}

# ==============================
# تشغيل محلي فقط (Render يستخدم Gunicorn/Uvicorn تلقائياً)
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
