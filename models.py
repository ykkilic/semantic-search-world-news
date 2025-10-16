from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine, func
from database import Base

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True)
    source = Column(String)
    title = Column(String)
    link = Column(String)
    summary = Column(String)
    content = Column(String)
    published = Column(DateTime)
    created_date= Column(DateTime)
    analyzed_news = Column(JSON)

class RSSFeed(Base):
    __tablename__ = "rss_feeds"
    
    id = Column(Integer, primary_key=True, index=True)
    rss_feed_name = Column(String(255), nullable=False)
    rss_feed_url = Column(String(512), nullable=False)