from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from settings import settings

Base = declarative_base()


class Database:
    def __init__(self, url: str):
        engine_kwargs = {}
        if url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(url, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    @contextmanager
    def session_scope(self) -> Generator:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        import database.models  # noqa: F401

        Base.metadata.create_all(bind=self.engine)

    def save_session(self, session_id: str, title: str, msg_cnt: Optional[int] = None) -> None:
        from database.models import ChatSession

        with self.session_scope() as session:
            record = session.get(ChatSession, session_id)
            if record is None:
                record = ChatSession(session_id=session_id, title=title, msg_cnt=msg_cnt or 0)
                session.add(record)
                return

            record.title = title
            if msg_cnt is not None:
                record.msg_cnt = msg_cnt

    def get_session(self, session_id: str) -> Optional[Dict]:
        from database.models import ChatSession

        with self.session_scope() as session:
            record = session.get(ChatSession, session_id)
            if record is None:
                return None
            return record.to_dict()

    def get_all_sessions(self) -> List[Dict]:
        from database.models import ChatSession

        with self.session_scope() as session:
            records = session.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
            return [record.to_dict() for record in records]

    def delete_session(self, session_id: str) -> None:
        from database.models import ChatSession

        with self.session_scope() as session:
            record = session.get(ChatSession, session_id)
            if record is not None:
                session.delete(record)

    def delete_all_sessions(self) -> None:
        from database.models import ChatMessage, ChatSession

        with self.session_scope() as session:
            session.query(ChatMessage).delete()
            session.query(ChatSession).delete()

    def save_message(self, session_id: str, role: str, content: str) -> None:
        from database.models import ChatMessage

        with self.session_scope() as session:
            session.add(ChatMessage(session_id=session_id, role=role, content=content))

    def get_messages(self, session_id: str) -> List[Dict]:
        from database.models import ChatMessage

        with self.session_scope() as session:
            records = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
                .all()
            )
            return [record.to_dict() for record in records]


db = Database(settings.DATABASE_URL)
