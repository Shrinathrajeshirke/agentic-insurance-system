import os
import uuid
import shutil
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.graph import agent_app
from src.agents.tools import ingest_policy_document
from src.schema.user_profile import UserProfile
from src.security.sanitizer import sanitize_user_profile_for_storage
from src.logger import logger
import uvicorn

app = FastAPI(title="Term Life Insurance Advisor API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[Message]
    user_profile: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    user_profile: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Ensure a valid thread_id exists for the checkpointer
        session_id = request.session_id if request.session_id else str(uuid.uuid4())

        # 2. Convert messages to LangChain types
        langchain_messages = []
        for msg in request.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        profile = UserProfile(**request.user_profile) if request.user_profile else UserProfile()

        # 3. Sanitize profile before logging
        audit_safe_profile = sanitize_user_profile_for_storage(profile.model_dump())
        logger.info(f"Session {session_id} - Processing request. Profile: {audit_safe_profile}")

        initial_state = {
            "messages": langchain_messages,
            "user_profile": profile,
            "retrieved_policies": [],
            "next_action": ""
        }

        # 4. Supply thread_id config to LangGraph checkpointer
        config = {"configurable": {"thread_id": session_id}}
        final_state = agent_app.invoke(initial_state, config=config)

        ai_reply = final_state["messages"][-1].content
        updated_profile = final_state["user_profile"].model_dump()

        return ChatResponse(reply=ai_reply, user_profile=updated_profile)

    except Exception as e:
        logger.error(f"Chat API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_brochure")
async def upload_brochure_endpoint(
    file: UploadFile = File(...),
    policy_name: str = Form(...),
    insurer: str = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF policy documents are supported.")

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingestion_status = ingest_policy_document(
            file_path=temp_file_path,
            policy_name=policy_name.strip(),
            insurer=insurer.strip()
        )
        return {"status": "success", "message": ingestion_status}
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)