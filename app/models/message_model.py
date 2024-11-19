from sqlalchemy import Column, Integer, String, Text, DateTime, func
from .database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=False)  # "user" ou "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Message(id={self.id}, sender={self.sender}, content={self.content[:20]}...)>"
