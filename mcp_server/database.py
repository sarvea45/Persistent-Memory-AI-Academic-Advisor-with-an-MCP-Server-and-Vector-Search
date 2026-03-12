import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import event

Base = declarative_base()

DB_PATH = os.environ.get("DB_PATH", "/app/data/advisor_memory.db")


class ConversationTable(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    turn_id = Column(Integer, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_conversations_user_turn", "user_id", "turn_id", unique=True),
    )


class UserPreferencesTable(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    preferences = Column(Text, nullable=False, default="{}")  # JSON string


class MilestoneTable(Base):
    __tablename__ = "milestones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    milestone_id = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    date_achieved = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_milestones_user_milestone", "user_id", "milestone_id", unique=True),
    )


def get_engine():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    return engine


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()