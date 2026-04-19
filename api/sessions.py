from fastapi import APIRouter
from fastapi import Response

from fastapi.responses import JSONResponse

from core.session_manager import session_manager
from database import db

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Mnafie Legal Group Chat API"}

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.head("/health")
async def health_head():
    return Response(status_code=200)

@router.post("/sessions")
async def create_session():
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@router.get("/sessions")
async def list_sessions():
    return {"sessions": session_manager.list_sessions()}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    try:
        messages = db.get_messages(session_id)
        return {"messages": messages}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        session_manager.delete_session(session_id)
        return {"message": "Session deleted"}
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@router.delete("/sessions")
async def delete_all_sessions():
    session_manager.delete_all_sessions()
    return {"message": "All sessions deleted"}
