from sqlalchemy import Column, Integer, String, DateTime, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./news.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    """
    Veritabanı oturumu (SessionLocal) oluşturur ve 'yield' ile sunar.
    İstek tamamlandığında (try bloğundan çıkıldığında) oturumu kapatmayı garanti eder.
    """
    db = SessionLocal()
    try:
        yield db  # Oturumu FastAPI'a teslim et
    finally:
        db.close() # Oturumu kapat

