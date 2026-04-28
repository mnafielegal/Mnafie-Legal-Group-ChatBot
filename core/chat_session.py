from datetime import datetime
from chatbot import LangChainChatBot
from tools import ChatbotResponse


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.title = "New Chat"
        self.agent = LangChainChatBot()
        self.message_count = 0


    def to_dict(self):
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
        }
    

    def generate_reply(self, message: str) -> ChatbotResponse:
        result = self.agent.agent_executor.invoke({"input": message})
        return result.get("output", "No response generated")
