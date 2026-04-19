import logging
import json
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from api.sessions import router as sessions_router
from core.session_manager import session_manager
from core.chat_request import ChatRequest
from database import db
from tools import rag_tool

logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)


@app.on_event("startup")
async def startup_event():
    db.create_tables()
    rag_tool.ensure_ready()


async def parse_chat_request(request: Request) -> ChatRequest:
    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(status_code=422, detail="Request body is required.")

    body_text = raw_body.decode("utf-8").strip()
    content_type = request.headers.get("content-type", "")

    data = None
    if "application/json" in content_type:
        try:
            data = json.loads(body_text, strict=False)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid JSON body: {exc.msg}",
            ) from exc
    elif "application/x-www-form-urlencoded" in content_type:
        parsed = parse_qs(body_text, keep_blank_values=True)
        data = {key: values[-1] for key, values in parsed.items()}
    else:
        try:
            data = json.loads(body_text, strict=False)
        except json.JSONDecodeError:
            parsed = parse_qs(body_text, keep_blank_values=True)
            if parsed:
                data = {key: values[-1] for key, values in parsed.items()}

    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Body must contain session_id and message.")

    try:
        chat_request = ChatRequest.model_validate(data)
        return chat_request
    except Exception as exc:
        raise HTTPException(status_code=422, detail="session_id and message are required.") from exc


@app.post("/chat")
async def chat(request: Request):
    chat_request = await parse_chat_request(request)
    if not chat_request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty.")

    session = None
    try:
        try:
            session = session_manager.open_session(chat_request.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        session_manager.update_session_title(chat_request.session_id, chat_request.message)
        session_manager.save_message(chat_request.session_id, "user", chat_request.message)

        try:
            response = session.generate_reply(chat_request.message)
        except Exception as exc:
            logger.exception("Failed to generate reply for session %s", chat_request.session_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        session_manager.save_message(chat_request.session_id, "assistant", response.reply)
        return {
            "session_id": chat_request.session_id,
            "reply": response.reply,
            "transfer": response.transfer,
            "transfer_to": response.transfer_to,
        }
    finally:
        if session is not None:
            session_manager.close_session(chat_request.session_id)
