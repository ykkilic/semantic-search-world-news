from bs4 import BeautifulSoup
import feedparser
import requests
from database import get_db, SessionLocal, Base, engine
from models import News
import dateutil.parser
from datetime import datetime
import pytz
from qdrant_service import index_news_to_qdrant
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import List, Dict
from models import RSSFeed
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

istanbul_tz = pytz.timezone('Europe/Istanbul')

# RSS_FEEDS = {
#     "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
#     "Hacker News": "https://news.ycombinator.com/rss",
#     "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
#     "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
#     "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
#     "guardian": "https://www.theguardian.com/world/rss",
#     "New York Times" : "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
#     "Al Jazeera" : "https://www.aljazeera.com/xml/rss/all.xml",
#     "TechCrunch" : "https://techcrunch.com/feed/",
#     "Wired" : "https://www.wired.com/feed/rss",
#     "Ars Technica" : "https://arstechnica.com/feed/",
#     "Nasa" : "https://www.nasa.gov/rss/dyn/breaking_news.rss"
# }

def get_all_rss_feeds_as_json() -> List[Dict]:
    """
    Veritabanındaki tüm RSS feed'lerini çekip, 
    istenilen formatta bir sözlükler listesi olarak döndürür.
    
    Args:
        db: SQLAlchemy oturum nesnesi.

    Returns:
        Her bir feed'i temsil eden sözlüklerin listesi.
        Örnek format: [{"rss_feed_name": "Yahoo Finance", "rss_feed_url": "..."}]
    """
    db = SessionLocal()
    try:
        feeds = db.execute(
            select(RSSFeed.rss_feed_name, RSSFeed.rss_feed_url)
        ).all()
        rss_feeds_dict = {} 
        for name, url in feeds:
            rss_feeds_dict[name] = url
            
        return rss_feeds_dict

    except Exception as e:
        print(f"Veritabanından RSS feed'leri çekilirken bir hata oluştu: {e}")
        return {}
    finally:
        db.close()

RSS_FEEDS = get_all_rss_feeds_as_json()

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
    logger.info(f"Fetch started at {datetime.now()}")
    db = SessionLocal()
    try:
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
        logger.info(f"Fetch finished at {datetime.now()}")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()

