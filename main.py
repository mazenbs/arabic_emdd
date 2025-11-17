
import re
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_PATH = "models/intfloat_multilingual-e5-small_merged_int8.onnx"
TOKENIZER_PATH = "models/tool"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

app = FastAPI(title="Arabic Text Embedding API")

==============================

دوال مساعدة

==============================

def normalize_arabic(text: str) -> str:
text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
text = re.sub(r"[إأآ]", "ا", text)
text = re.sub(r"ى", "ي", text)
text = re.sub(r"ؤ", "و", text)
text = re.sub(r"ئ", "ي", text)
text = re.sub(r"ة", "ه", text)
text = re.sub(r"[^\w\s]", " ", text)
text = re.sub(r"\s+", " ", text)
return text.strip()

def split_sentences(text: str):
sentences = re.split(r"[.\n:؛؟!]", text)
return [s.strip() for s in sentences if len(s.strip()) > 0]

def embed_text(text: str):
text = normalize_arabic(text)
sentences = split_sentences(text)
if not sentences:
return None

all_vectors = []  
for s in sentences:  
    input_text = "passage: " + s  
    inputs = tokenizer(input_text, return_tensors="np", truncation=True, max_length=256)  
    ort_inputs = {k: v for k, v in inputs.items()}  
    ort_outs = session.run(None, ort_inputs)  
    vector = ort_outs[0][0]  
    vector = vector / (np.linalg.norm(vector) + 1e-10)  
    all_vectors.append(vector)  

# المتوسط = embedding واحد نهائي  
embedding = np.mean(np.stack(all_vectors, axis=0), axis=0)  
embedding = embedding / (np.linalg.norm(embedding) + 1e-10)  
return embedding.astype(np.float32)

==============================

نموذج الإدخال

==============================

class TextRequest(BaseModel):
text: str

==============================

المسارات

==============================

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
embedding = embed_text(request.text)
if embedding is None:
raise HTTPException(status_code=400, detail="لم يتم العثور على جمل صالحة في النص.")
return {"embedding": embedding.tolist()}

==============================

تشغيل محلي

==============================

if name == "main":
import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000)