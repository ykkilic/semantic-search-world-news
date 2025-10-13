import feedparser
from fastapi import FastAPI, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import dateutil.parser
from bs4 import BeautifulSoup
import requests
import pytz
from database import get_db, SessionLocal
from models import News
from qdrant_service import index_news_to_qdrant, semantic_search

istanbul_tz = pytz.timezone('Europe/Istanbul')

app = FastAPI(title="RSS News API with SQLite")

RSS_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Hacker News": "https://news.ycombinator.com/rss",
    "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "guardian": "https://www.theguardian.com/world/rss",
    "New York Times" : "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Al Jazeera" : "https://www.aljazeera.com/xml/rss/all.xml",
    "TechCrunch" : "https://techcrunch.com/feed/",
    "Wired" : "https://www.wired.com/feed/rss",
    "Ars Technica" : "https://arstechnica.com/feed/",
    "Nasa" : "https://www.nasa.gov/rss/dyn/breaking_news.rss"
}

def clean_summary(html_summary):
    if not html_summary:
        return ""
    soup = BeautifulSoup(html_summary, "html.parser")
    return soup.get_text()

def fetch_full_article(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs)
            return text.strip()
        return ""
    except Exception:
        return ""


def fetch_and_store():
    db = SessionLocal()
    istanbul_timestamp = datetime.now(istanbul_tz)
    for source, feed_url in RSS_FEEDS.items():
        parsed_feed = feedparser.parse(feed_url)
        for entry in parsed_feed.entries:
            try:
                published = dateutil.parser.parse(entry.get("published"))
            except:
                published = None

            exists = db.query(News).filter(News.link == entry.get("link")).first()
            if not exists:
                full_content = fetch_full_article(entry.get("link"))

                news_item = News(
                    source=source,
                    title=entry.get("title"),
                    link=entry.get("link"),
                    content=full_content,  
                    published=published,
                    created_date=istanbul_timestamp
                )
                db.add(news_item)
    db.commit()
    index_news_to_qdrant()



@app.get("/")
def get_all_articles(db: Session = Depends(get_db)):
    
    latest_news = db.query(News).order_by(News.created_date.desc()).first()
    now = datetime.now(istanbul_tz)

    if latest_news:
        created = latest_news.created_date
        if created.tzinfo is None:
            created = istanbul_tz.localize(created)
    else:
        created = None

    if not latest_news or (now - created > timedelta(minutes=15)):
        fetch_and_store()

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
    
    latest_news = db.query(News).order_by(News.created_date.desc()).first()
    now = datetime.now(istanbul_tz)

    if latest_news:
        created = latest_news.created_date
        if created.tzinfo is None:
            created = istanbul_tz.localize(created)
    else:
        created = None

    if not latest_news or (now - created > timedelta(minutes=15)):
        fetch_and_store()

    results = db.query(News).filter(News.title.ilike(f"%{q}%")).all()
    return {
        "query": q,
        "results": [
            {
                "source": article.source,
                "title": article.title,
                "link": article.link,
                "summary": article.content,
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
    return {"sources": list(RSS_FEEDS.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
