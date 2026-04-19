from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    attachment_type: str | None = None
    attachment_url: str | None = None
