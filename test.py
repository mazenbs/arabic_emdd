# test_albert_onnx.py
import onnxruntime as ort
from transformers import AlbertTokenizer
import numpy as np

# ----------------------------
# تحميل النموذج والـ tokenizer
# ----------------------------
TOKENIZER_PATH = "models/asafaya/albert-base-arabic"
MODEL_PATH = "models/albert_arabic_wa_merged.onnx"
TARGET_DIM = 384

print("🔹 تحميل Tokenizer ...")
tokenizer = AlbertTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)

print("🔹 تحميل نموذج ONNX ...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ----------------------------
# إعداد مصفوفة إسقاط لتقليل الأبعاد
# ----------------------------
np.random.seed(42)
projection_matrix = np.random.randn(768, TARGET_DIM).astype(np.float32)

# ----------------------------
# دالة تحويل النص إلى متجه
# ----------------------------
def get_embedding(text: str):
    if not text.strip():
        raise ValueError("⚠️ النص فارغ!")

    # Tokenize
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # استنتاج ONNX
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    embedding_768 = outputs[0].mean(axis=1)  # متوسط كل token

    # إسقاط إلى 384
    embedding_384 = embedding_768 @ projection_matrix
    return embedding_384[0]

# ----------------------------
# اختبار تفاعلي
# ----------------------------
if __name__ == "__main__":
    print("✅ تم تحميل النموذج بنجاح!")
    while True:
        text = input("\n📝 أدخل نصًا بالعربية (أو اكتب 'خروج' لإنهاء): ").strip()
        if text.lower() in ["خروج", "exit", "quit"]:
            print("👋 تم الإنهاء.")
            break
        try:
            embedding = get_embedding(text)
            print(f"\n🔸 المتجه الناتج ({len(embedding)} بعد):")
            print(embedding)
        except Exception as e:
            print(f"❌ خطأ: {e}")
