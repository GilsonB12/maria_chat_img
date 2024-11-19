import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://chat_user:chat_password@localhost:5433/chat_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def initialize_database():
    """Inicializa o banco de dados e cria tabelas."""
    from app.models.message_model import Message
    Base.metadata.create_all(bind=engine)
