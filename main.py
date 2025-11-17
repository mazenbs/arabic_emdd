import re
import logging
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

# المسارات إلى النموذج والمُجمّع (تأكد أنها صحيحة في بيئتك)
MODEL_PATH = "models/intfloat_multilingual-e5-small_merged_int8.onnx"
TOKENIZER_PATH = "models/tool"

# تهيئة سجل الأخطاء
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة جلسة ONNX والـ tokenizer
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

app = FastAPI(title="Arabic Text Embedding API")


# ==============================
# دوال مساعدة
# ==============================

def normalize_arabic(text: str) -> str:
    # إزالة التشكيل وتوحيد بعض الحروف واستبدال الرموز غير المرغوبة
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    # استبدال أي شيء ليس كلمة أو مسافة بمسافة
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    # تقطيع نص إلى جمل على أساس علامات الترقيم العربية/الإنجليزية والأسطر الجديدة
    sentences = re.split(r"[.\n:؛؟!]", text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def embed_text(text: str) -> Optional[np.ndarray]:
    # تطبيع وتقسيم
    text = normalize_arabic(text)
    sentences = split_sentences(text)
    if not sentences:
        return None

    all_vectors: List[np.ndarray] = []

    for s in sentences:
        input_text = "passage: " + s
        # نطلب مخرجات كـ numpy arrays
        inputs = tokenizer(input_text, return_tensors="np", truncation=True, max_length=256)
        # ONNX Runtime يتوقع numpy arrays كقيم
        ort_inputs = {k: v for k, v in inputs.items()}

        try:
            ort_outs = session.run(None, ort_inputs)
        except Exception as e:
            logger.exception("ONNX Runtime failed to run")
            raise

        # افتراض أن المخرج الأول هو متجه واحد للشكل (1, dim) أو (batch, dim)
        vector = ort_outs[0][0]
        # تطبيع المتجه
        norm = np.linalg.norm(vector)
        if norm == 0:
            logger.warning("Zero norm encountered for a sentence embedding.")
            normalized = vector.astype(np.float32)
        else:
            normalized = (vector / (norm + 1e-10)).astype(np.float32)

        all_vectors.append(normalized)

    # حساب المتوسط على مستوى الجمل لإنتاج embedding واحد للنص كله
    embedding = np.mean(np.stack(all_vectors, axis=0), axis=0)
    emb_norm = np.linalg.norm(embedding)
    if emb_norm > 0:
        embedding = embedding / (emb_norm + 1e-10)
    return embedding.astype(np.float32)


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
    return {"message": "✅ Arabic Text Embedding API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed_endpoint(request: TextRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ.")

    try:
        embedding = embed_text(request.text)
    except Exception as e:
        logger.exception("Failed to create embedding")
        raise HTTPException(status_code=500, detail="فشل في إنشاء التضمين (embedding).")

    if embedding is None:
        raise HTTPException(status_code=400, detail="لم يتم العثور على جمل صالحة في النص.")

    return {"embedding": embedding.tolist()}


# ==============================
# تشغيل محلي
# ==============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)