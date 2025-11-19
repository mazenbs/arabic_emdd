# main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from embeddingonnx import text_to_embedding, query_to_embedding  # استيراد الدوال المعدة مسبقًا

# ==============================
# إنشاء تطبيق FastAPI
# ==============================
app = FastAPI(title="Arabic Text Embedding API")

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

# ==============================
# نقاط النهاية الأصلية POST
# ==============================
@app.post("/embed")
def embed_endpoint(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="النص فارغ.")
    try:
        vector = text_to_embedding(text, normalize=True)
        if vector is None:
            raise HTTPException(status_code=400, detail="لم يتم إنشاء embedding للنص.")
        return {"embedding": vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء إنشاء embedding: {str(e)}")

@app.post("/query")
def query_endpoint(request: TextRequest):
    query_text = request.text.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="النص فارغ.")
    try:
        vector = query_to_embedding(query_text, normalize=True)
        if vector is None:
            raise HTTPException(status_code=400, detail="لم يتم إنشاء embedding للاستعلام.")
        return {"query_embedding": vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء إنشاء embedding للاستعلام: {str(e)}")

# ==============================
# نقاط النهاية الجديدة GET
# ==============================
@app.get("/embed")
def embed_get(text: str = Query(..., description="النص المراد تحويله إلى embedding")):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="النص فارغ.")
    try:
        vector = text_to_embedding(text, normalize=True)
        if vector is None:
            raise HTTPException(status_code=400, detail="لم يتم إنشاء embedding للنص.")
        return {"embedding": vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء إنشاء embedding: {str(e)}")

@app.get("/query")
def query_get(text: str = Query(..., description="النص المراد تحويله إلى query embedding")):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="النص فارغ.")
    try:
        vector = query_to_embedding(text, normalize=True)
        if vector is None:
            raise HTTPException(status_code=400, detail="لم يتم إنشاء embedding للاستعلام.")
        return {"query_embedding": vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء إنشاء embedding للاستعلام: {str(e)}")

# ==============================
# تشغيل السيرفر
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
