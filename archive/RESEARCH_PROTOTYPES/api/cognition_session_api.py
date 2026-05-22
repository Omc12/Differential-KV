from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Cognition Session API", version="1.0")

class SessionCreate(BaseModel):
    session_id: str
    config: Dict[str, Any]

class SessionStatus(BaseModel):
    session_id: str
    status: str
    entropy: float

active_sessions = {}

@app.post("/sessions/", response_model=SessionStatus)
async def create_session(session: SessionCreate):
    if session.session_id in active_sessions:
        raise HTTPException(status_code=400, detail="Session already exists")
    
    # Init mock session
    active_sessions[session.session_id] = {
        "status": "running",
        "entropy": 0.05,
        "config": session.config
    }
    return {"session_id": session.session_id, "status": "running", "entropy": 0.05}

@app.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = active_sessions[session_id]
    return {"session_id": session_id, "status": s["status"], "entropy": s["entropy"]}

@app.delete("/sessions/{session_id}")
async def terminate_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"message": f"Session {session_id} terminated."}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
