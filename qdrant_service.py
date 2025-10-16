from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from tqdm import tqdm 
import time
from models import News
from database import SessionLocal

# ---------------- Embedding Model ----------------
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

def get_embedding(text: str):
    return model.encode(text).tolist()

# ---------------- Qdrant Client ----------------
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=1800.0)
collection_name = "news_articles"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={"content": VectorParams(size=768, distance="Cosine")}
    )

# # ---------------- Indexleme ----------------
# def index_news_to_qdrant():
#     db: Session = SessionLocal()
#     try:
#         articles = db.query(News).all()
#         points = []
#         for article in articles:
#             if not article.content:
#                 continue
#             vector = get_embedding(article.content)
#             points.append({
#                 "id": article.id,
#                 "vector": {"content": vector},  
#                 "payload": {
#                     "title": article.title,
#                     "link": article.link,
#                     "source": article.source,
#                     "published": article.published.isoformat() if article.published else None,
#                     "content" : article.content
#                 }
#             })

#         if points:
#             qdrant_client.upsert(
#                 collection_name=collection_name,
#                 points=points
#             )
#         print(f"{len(points)} articles indexed to Qdrant.")
#     finally:
#         db.close()

BATCH_SIZE = 100  # 100-200 ideal; Qdrant hızına göre ayarlanabilir

def index_news_to_qdrant():
    db: Session = SessionLocal()
    
    try:
        total_articles = db.query(News).count()
        print(f"📰 Total {total_articles} articles found.")
        
        offset = 0
        indexed_count = 0
        
        # döngüyle parça parça çek
        while True:
            articles = (
                db.query(News)
                .order_by(News.id)
                .offset(offset)
                .limit(BATCH_SIZE)
                .all()
            )
            if not articles:
                break

            points = []
            for article in articles:
                if not article.content:
                    continue
                try:
                    vector = get_embedding(article.content)
                except Exception as e:
                    print(f"⚠️ Embedding failed for ID {article.id}: {e}")
                    continue

                points.append({
                    "id": article.id,
                    "vector": {"content": vector},
                    "payload": {
                        "title": article.title,
                        "link": article.link,
                        "source": article.source,
                        "published": article.published.isoformat() if article.published else None,
                        "content": article.content
                    }
                })

            if points:
                # Qdrant’a küçük batch’lerle gönder
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    indexed_count += len(points)
                    print(f"✅ Indexed batch of {len(points)} (total: {indexed_count})")
                except Exception as e:
                    print(f"❌ Qdrant upsert failed at offset {offset}: {e}")
                    # küçük bir bekleme (örneğin ağ yavaşsa)
                    time.sleep(5)

            offset += BATCH_SIZE

        print(f"🎯 Done. Total indexed: {indexed_count}")

    finally:
        db.close()

def semantic_search(query: str, top_k: int = 5):
    query_vector = model.encode(query).tolist()

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=("content", query_vector), 
        limit=top_k,
    )

    results = []
    for hit in search_result:
        if hit.score < 0.30:
            continue
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

