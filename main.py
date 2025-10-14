import threading
import time
from fastapi import FastAPI, Query, Depends
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import pytz
from database import get_db, SessionLocal, Base, engine
from models import News
from qdrant_service import semantic_search
from rss_service import fetch_and_store, get_all_rss_feeds_as_json

istanbul_tz = pytz.timezone('Europe/Istanbul')

# def run_periodically(interval: int):
#     db = SessionLocal()
#     try:
#         latest_news = db.query(News).order_by(News.created_date.desc()).first()
#         now = datetime.now(istanbul_tz)

#         if latest_news:
#             created = latest_news.created_date
#             if created.tzinfo is None:
#                 created = istanbul_tz.localize(created)
#         else:
#             created = None

#         if not latest_news or (now - created > timedelta(minutes=15)):
#             print("Yukarıda ki ifin içinde")
#             fetch_and_store()
#         print("While dan önce")
#         while True:
#             print("Çalıştı")
#             fetch_and_store()
#             print("Bitti")
#             time.sleep(interval)
#     except Exception as e:
#         print(e)
#         raise e
#     finally:
#         db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    # thread = threading.Thread(target=run_periodically, args=(900,), daemon=True)
    # thread.start()
    yield

app = FastAPI(title="RSS News API with SQLite", lifespan=lifespan)


@app.get("/")
def get_all_articles(db: Session = Depends(get_db)):

    articles = (
        db.query(News)
        .order_by(News.published.desc())
    )

    return {
        "articles": [
            {
                "source": article.source,
                "title": article.title,
                "link": article.link,
                "content": article.content,
                "published": article.published.isoformat() if article.published else None
            } for article in articles
        ]
    }


@app.get("/search")
def search_articles(q: str = Query(..., description="Aranacak kelime"), db: Session = Depends(get_db)):
    results = db.query(News).filter(News.title.ilike(f"%{q}%")).all()
    return {
        "query": q,
        "results": [
            {
                "source": article.source,
                "title": article.title,
                "link": article.link,
                "content": article.content,
                "published": article.published.isoformat() if article.published else None
            } for article in results
        ]
    }

@app.get("/semantic-search")
async def s_search(q: str = Query(..., description="Aranacak kelime"), db: Session = Depends(get_db)):
    try:
        result = semantic_search(q, top_k=10)
        return result
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={"message" : "Internal Server Error"}
        )

@app.get("/sources")
def list_sources():
    rss_feeds = get_all_rss_feeds_as_json()
    return {"sources": list(rss_feeds.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
