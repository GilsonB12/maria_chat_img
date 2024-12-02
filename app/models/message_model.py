from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, func
from .database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=True)  # "user" ou "assistant"
    content = Column(Text, nullable=True)
    image_path = Column(String, nullable=True)  # Caminho da imagem, se houver
    image_embedding = Column(JSON, nullable=True)  # Embedding da imagem
    timestamp = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Message(id={self.id}, sender={self.sender}, content={self.content[:20]}...)>"
