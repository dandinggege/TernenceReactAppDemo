from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import os
import uvicorn

# ------------------------------
# ⚠️ 本地模型路径，绝对路径
MODEL_PATH = "./all-distilroberta-v1"
# ------------------------------

# ⚠️ 禁止访问 HuggingFace 网络
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ------------------------------
# 初始化 FastAPI
app = FastAPI(title="Offline Embedding Service")

# ------------------------------
# 加载本地模型（CPU 可用）
print(f"Loading embedding model from {MODEL_PATH} ...")
model = SentenceTransformer(MODEL_PATH)
print("Model loaded successfully.")

# ------------------------------
# 请求和响应格式
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int

# ------------------------------
# embedding 接口
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vectors = model.encode(
        req.texts,
        batch_size=32,
        normalize_embeddings=True  # cosine 相似度搜索推荐
    )
    return {"embeddings": vectors.tolist(), "dim": vectors.shape[1]}

# ------------------------------
# 健康检查接口
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------
# main 启动方法
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087, reload=False)
