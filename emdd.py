import onnxruntime as ort
from transformers import AlbertTokenizer
import numpy as np

# ==============================
# الإعدادات العامة
# ==============================
TOKENIZER_PATH = "models/asafaya/albert-base-arabic"
MODEL_PATH = "models/albert_arabic_wa_merged.onnx"
TARGET_DIM = 384
CHUNK_SIZE = 150  # عدد الكلمات لكل chunk

# ==============================
# تحميل النموذج والمحول
# ==============================
tokenizer = AlbertTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(MODEL_PATH, sess_options=session_options, providers=["CPUExecutionProvider"])

# ==============================
# مصفوفة إسقاط لتقليل الأبعاد
# ==============================
np.random.seed(42)
projection_matrix = np.random.normal(0, 0.1, (768, TARGET_DIM)).astype(np.float32)

# ==============================
# تقسيم النص إلى مقاطع (chunks)
# ==============================
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# ==============================
# تحويل النص إلى متجه (Embedding)
# ==============================
def text_to_vector(
    text: str,
    mean_pooling: bool = True,
    normalize: bool = True,
    return_dim: int = TARGET_DIM
):
    if not text.strip():
        raise ValueError("❌ النص فارغ")

    chunks = chunk_text(text)
    all_embeddings = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="np", truncation=True, max_length=128)
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"]

        outputs = session.run(None, ort_inputs)
        last_hidden = outputs[0]

        if mean_pooling:
            embedding = last_hidden.mean(axis=1)
        else:
            embedding = last_hidden[:, 0, :]  # CLS token

        # الإسقاط لتقليل الأبعاد
        embedding_projected = embedding @ projection_matrix[:, :return_dim]

        # التطبيع
        if normalize:
            norm = np.linalg.norm(embedding_projected, axis=1, keepdims=True)
            embedding_projected = embedding_projected / (norm + 1e-10)

        all_embeddings.append(embedding_projected[0])

    # دمج كل المقاطع في متجه واحد
    final_embedding = np.mean(np.stack(all_embeddings), axis=0)
    if normalize:
        final_embedding /= (np.linalg.norm(final_embedding) + 1e-10)

    return final_embedding

# ==============================
# مثال للاستخدام
# ==============================
if __name__ == "__main__":
    text = "القانون اليمني من أفضل القوانين العربية"
    embedding = text_to_vector(text)
    print("✅ عدد الأبعاد:", len(embedding))
    print("📊 أول 10 قيم من المتجه:", embedding[:10])
