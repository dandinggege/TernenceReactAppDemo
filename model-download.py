from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-distilroberta-v1"

# 第一次会自动下载并缓存
model = SentenceTransformer(model_name)

print("Model downloaded successfully.")
