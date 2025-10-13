from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from models import News
from database import SessionLocal

# ---------------- Embedding Model ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()

# ---------------- Qdrant Client ----------------
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "news_articles"

# Collection yoksa oluştur
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={"content": VectorParams(size=384, distance="Cosine")}
    )

# ---------------- Indexleme ----------------
def index_news_to_qdrant():
    db: Session = SessionLocal()
    try:
        articles = db.query(News).all()
        points = []
        for article in articles:
            if not article.content:
                continue
            vector = get_embedding(article.content)
            points.append({
                "id": article.id,
                "vector": {"content": vector},  # <<< burada vector_name kullanımı
                "payload": {
                    "title": article.title,
                    "link": article.link,
                    "source": article.source,
                    "published": article.published.isoformat() if article.published else None,
                    "content" : article.content
                }
            })

        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
        print(f"{len(points)} articles indexed to Qdrant.")
    finally:
        db.close()
    
def semantic_search(query: str, top_k: int = 5):
    query_vector = model.encode(query).tolist()

    # Search API: vector_name ve vector belirtilmeli
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=("content", query_vector), 
        limit=top_k,
    )

    results = []
    for hit in search_result:
        # hit: ScoredPoint
        results.append({
            "id": hit.id,
            "score": hit.score,
            "title": hit.payload.get("title"),
            "link": hit.payload.get("link"),
            "source": hit.payload.get("source"),
            "published": hit.payload.get("published"),
            "content" : hit.payload.get("content")
        })
    return results

# if __name__ == "__main__":
#     index_news_to_qdrant()

# ---------------- Test ----------------
# if __name__ == "__main__":
#     query = "Ukraine"
#     results = semantic_search(query, top_k=10)
#     for i, r in enumerate(results, 1):
#         print(f"{i}. {r['title']} ({r['score']:.4f}) - {r['link']}")
#         print(f"{i}. {r['content']}")

