import onnxruntime as ort
import re
import numpy as np
from transformers import AutoTokenizer

# ==============================
# إعداد النموذج والتوكنيزر
# ==============================
MODEL_PATH = "models/intfloat_multilingual-e5-small_merged_int8.onnx"
TOKENIZER_PATH = "models/tool"

# جلسة ONNX
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# توكنيزر محلي (بدون تحميل من الإنترنت)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# ==============================
# دوال مساعدة
# ==============================
def normalize_arabic(text: str) -> str:
    """
    تطبيع كامل للنص العربي عند normalize=True
    """
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)          # إزالة التشكيل
    text = re.sub(r'[إأآ]', 'ا', text)              # توحيد الهمزات
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة\b', 'ه', text)                # ة → ه آخر الكلمة
    text = re.sub(r'[^\w\s]', ' ', text)            # إزالة الرموز
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

    
def split_sentences(text: str):
    sentences = re.split(r'[.\n:؛؟!]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# ==============================
# الدالة الرئيسية لتحويل النص إلى متجه
# ==============================
def text_to_embedding(text: str, normalize: bool = True) -> np.ndarray:
    """
    تحويل نص إلى متجه واحد صالح للتخزين في Supabase.
    تتعامل مع النصوص الطويلة عن طريق تقسيمها إلى جمل وأخذ متوسط متجهاتها.
    
    Args:
        text (str): النص العربي المراد تحويله.
        normalize (bool): إذا كان True، سيتم تطبيع النص العربي قبل التحويل.
    
    Returns:
        np.ndarray: متجه 1D بحجم 384.
    """
    if normalize:
        text = normalize_arabic(text)

    # تقسيم النص إلى جمل
    sentences = split_sentences(text)
    if not sentences:
        return None

    vectors = []
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

        # استخدام CLS pooled output (384D)
        vector = ort_outs[1][0]
        vector = vector / (np.linalg.norm(vector) + 1e-10)
        vectors.append(vector)

    # المتوسط لإنتاج متجه واحد 1D
    embedding = np.mean(np.stack(vectors, axis=0), axis=0)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

    return embedding.astype(np.float32)

def query_to_embedding(query: str, normalize: bool = True) -> np.ndarray:
    """
    إنشاء embedding لعملية البحث باستخدام 'query:' 
    وتستخدم لاستعلامات المستخدم فقط.
    """
    if not query.strip():
        return None

    # تطبيع النص (اختياري)
    if normalize:
        query = normalize_arabic(query)

    # تجهيز النص للنموذج
    input_text = "query: " + query

    # تحويل التوكنيزر إلى مدخلات ONNX
    inputs = tokenizer(
        input_text,
        return_tensors="np",
        truncation=True,
        max_length=256
    )

    ort_inputs = {k: v for k, v in inputs.items()}

    # تشغيل النموذج
    ort_outs = session.run(None, ort_inputs)

    # استخدام CLS pooled output (الأفضل للبحث)
    vector = ort_outs[1][0]     # (384D)

    # تطبيع المتجه
    vector = vector / (np.linalg.norm(vector) + 1e-10)

    return vector.astype(np.float32)
