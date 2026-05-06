from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

# Text queries
queries = [
    "A woman playing with her dog on a beach at sunset.",
    "Pet owner training dog outdoors near water.",
    "Woman surfing on waves during a sunny day.",
    "City skyline view from a high-rise building at night.",
]

# Documents: text, image, and text+image
documents = [
    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    {
        "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    },
]

# Encode queries and documents
query_embeddings = model.encode(queries)
doc_embeddings = model.encode(documents)
print(query_embeddings.shape, doc_embeddings.shape)
# (4, 2048) (3, 2048)

# Compute similarities
similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)
# tensor([[0.8160, 0.7155, 0.7054],
#         [0.5173, 0.3295, 0.4446],
#         [0.3863, 0.2987, 0.3312],
#         [0.1061, 0.0433, 0.0839]])
