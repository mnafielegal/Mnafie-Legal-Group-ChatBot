import uuid
from datetime import datetime
from typing import Dict

from core.chat_session import Session
from database import db


class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}
        self.active_session_connections: Dict[str, int] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        db.save_session(session_id, "New Chat")
        return session_id

    def _build_session_from_db(self, session_id: str) -> Session:
        session_data = db.get_session(session_id)
        if session_data is None:
            raise ValueError(f"Session {session_id} not found in database")

        session = Session(session_id)

        if session_data:
            session.title = session_data.get("title", "New Chat")
            session.message_count = session_data.get("msg_cnt", 0)
            if session_data.get("created_at"):
                session.created_at = datetime.fromisoformat(session_data["created_at"])
            if session_data.get("updated_at"):
                session.updated_at = datetime.fromisoformat(session_data["updated_at"])

        messages = db.get_messages(session_id)
        for msg in messages:
            if msg["role"] == "user":
                session.agent.memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                session.agent.memory.add_ai_message(msg["content"])

        return session

    def open_session(self, session_id: str) -> Session:
        session = self.active_sessions.get(session_id)
        if session is None:
            session = self._build_session_from_db(session_id)
            self.active_sessions[session_id] = session
            self.active_session_connections[session_id] = 0

        self.active_session_connections[session_id] += 1
        return session

    def get_session(self, session_id: str) -> Session:
        session = self.active_sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} is not active")
        return session

    def close_session(self, session_id: str):
        connection_count = self.active_session_connections.get(session_id)
        if connection_count is None:
            return

        if connection_count <= 1:
            self.active_session_connections.pop(session_id, None)
            self.active_sessions.pop(session_id, None)
            return

        self.active_session_connections[session_id] = connection_count - 1

    def delete_session(self, session_id: str):
        session_data = db.get_session(session_id)
        if session_data is None:
            raise ValueError(f"Session {session_id} not found in database")

        self.active_sessions.pop(session_id, None)
        self.active_session_connections.pop(session_id, None)
        db.delete_session(session_id)

    def delete_all_sessions(self):
        self.active_sessions.clear()
        self.active_session_connections.clear()
        db.delete_all_sessions()

    def list_sessions(self):
        return db.get_all_sessions()

    def update_session_title(self, session_id: str, first_message: str):
        session_data = db.get_session(session_id)
        if session_data is None:
            raise ValueError(f"Session {session_id} not found in database")

        message_count = session_data.get("msg_cnt", 0)
        title = session_data.get("title", "New Chat")
        if message_count == 0:
            title = first_message[:50] + "..." if len(first_message) > 50 else first_message

        message_count += 1
        db.save_session(session_id, title, msg_cnt=message_count)

        session = self.active_sessions.get(session_id)
        if session is not None:
            session.title = title
            session.message_count = message_count
            session.updated_at = datetime.now()

    def save_message(self, session_id: str, role: str, content: str):
        db.save_message(session_id, role, content)


session_manager = SessionManager()
