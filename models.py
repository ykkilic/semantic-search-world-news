from sqlalchemy import Column, Integer, String, DateTime, create_engine, func
from database import Base

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    title = Column(String, index=True)
    link = Column(String)
    summary = Column(String)
    content = Column(String)
    published = Column(DateTime, index=True)
    created_date= Column(DateTime, index=True)

class RSSFeed(Base):
    __tablename__ = "rss_feeds"
    
    id = Column(Integer, primary_key=True, index=True)
    rss_feed_name = Column(String(255), nullable=False)
    rss_feed_url = Column(String(512), nullable=False)