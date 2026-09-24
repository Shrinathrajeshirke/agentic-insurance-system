import os
import uuid
import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from fastapi.responses import Response
from src.tools.report_generator import generate_pdf_advisory_report
from src.agents.graph import workflow
from src.agents.tools import ingest_policy_document
from src.schema.user_profile import UserProfile
from src.security.sanitizer import sanitize_user_profile_for_storage
from src.security.auth import hash_password, verify_password, create_access_token, get_current_user
from src.db.models import get_db, User, ChatThread
from src.logger import logger
import uvicorn

# Global compiled agent instance
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    async with AsyncSqliteSaver.from_conn_string("conversations.db") as checkpointer:
        agent_app = workflow.compile(checkpointer=checkpointer)
        logger.info("LangGraph compiled successfully with AsyncSqliteSaver.")
        yield

app = FastAPI(title="Term Life Insurance Advisor API", lifespan=lifespan)

# --- Schemas ---
class UserAuth(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str

class ThreadItem(BaseModel):
    id: str
    title: str
    created_at: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    thread_id: str
    messages: List[Message]
    user_profile: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    user_profile: Dict[str, Any]

# --- Auth Endpoints ---
@app.post("/auth/register", response_model=TokenResponse)
def register(auth_data: UserAuth, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == auth_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    new_user = User(
        email=auth_data.email,
        hashed_password=hash_password(auth_data.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(data={"sub": new_user.id, "email": new_user.email})
    return TokenResponse(access_token=token, email=new_user.email)

@app.post("/auth/login", response_model=TokenResponse)
def login(auth_data: UserAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == auth_data.email).first()
    if not user or not verify_password(auth_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    
    token = create_access_token(data={"sub": user.id, "email": user.email})
    return TokenResponse(access_token=token, email=user.email)

# --- Thread Endpoints ---
@app.get("/threads", response_model=List[ThreadItem])
def get_user_threads(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    threads = db.query(ChatThread).filter(ChatThread.user_id == current_user.id).order_by(ChatThread.created_at.desc()).all()
    return [
        ThreadItem(
            id=t.id, 
            title=t.title, 
            created_at=t.created_at.strftime("%b %d, %H:%M")
        ) for t in threads
    ]

@app.post("/threads/new", response_model=ThreadItem)
def create_thread(
    title: Optional[str] = "New Advisory Session", 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    new_thread = ChatThread(user_id=current_user.id, title=title)
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return ThreadItem(
        id=new_thread.id, 
        title=new_thread.title, 
        created_at=new_thread.created_at.strftime("%b %d, %H:%M")
    )

class ThreadRenameRequest(BaseModel):
    title: str

@app.patch("/threads/{thread_id}/rename", response_model=ThreadItem)
def rename_thread(
    thread_id: str,
    body: ThreadRenameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id,
        ChatThread.user_id == current_user.id
    ).first()

    if not thread:
        raise HTTPException(status_code=404, detail="Session not found.")

    new_title = body.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    thread.title = new_title
    db.commit()
    db.refresh(thread)

    return ThreadItem(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at.strftime("%b %d, %H:%M")
    )

@app.delete("/threads/{thread_id}")
def delete_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id,
        ChatThread.user_id == current_user.id
    ).first()

    if not thread:
        raise HTTPException(status_code=404, detail="Session not found.")

    db.delete(thread)
    db.commit()

    return {"status": "success", "message": "Session deleted."}

# --- Async Streaming Chat Endpoint ---
@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if agent_app is None:
        raise HTTPException(status_code=503, detail="Agent workflow is not initialized.")

    thread = db.query(ChatThread).filter(
        ChatThread.id == request.thread_id,
        ChatThread.user_id == current_user.id
    ).first()

    if not thread:
        thread = ChatThread(id=request.thread_id, user_id=current_user.id, title="Advisory Session")
        db.add(thread)
        db.commit()

    if thread.title == "New Advisory Session" and len(request.messages) > 0:
        first_user_msg = next((m.content for m in request.messages if m.role == "user"), None)
        if first_user_msg and "initial profile assessment" not in first_user_msg.lower():
            thread.title = first_user_msg[:30] + ("..." if len(first_user_msg) > 30 else "")
            db.commit()

    langchain_messages = []
    for msg in request.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    profile = UserProfile(**request.user_profile) if request.user_profile else UserProfile()
    audit_safe = sanitize_user_profile_for_storage(profile.model_dump())
    logger.info(f"Streaming request | User: {current_user.email} | Thread: {request.thread_id} | Profile: {audit_safe}")

    initial_state = {
        "messages": langchain_messages,
        "user_profile": profile,
        "retrieved_policies": [],
        "next_action": ""
    }

    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_generator():
        try:
            citations_sent = False
            async for event in agent_app.astream_events(initial_state, config=config, version="v2"):
                event_type = event.get("event")
                
                # Emit citations once retriever finishes
                if not citations_sent and event_type == "on_chain_end" and event.get("name") == "retriever":
                    output = event.get("data", {}).get("output", {})
                    retrieved = output.get("retrieved_policies", [])
                    clean_citations = [
                        {
                            "insurer": r.get("insurer"),
                            "policy_name": r.get("policy_name"),
                            "page": r.get("page", "N/A"),
                            "section": r.get("section", "Contract Clause"),
                            "snippet": r.get("text", "")[:350] + ("..." if len(r.get("text", "")) > 350 else "")
                        }
                        for r in retrieved if r.get("insurer") != "External Web Result"
                    ]
                    if clean_citations:
                        yield {"data": json.dumps({"citations": clean_citations})}
                    citations_sent = True

                # Stream tokens from advisor
                if event_type == "on_chat_model_stream":
                    metadata = event.get("metadata", {})
                    if metadata.get("langgraph_node") == "advisor":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and chunk.content:
                            yield {"data": json.dumps({"token": chunk.content})}

            yield {"data": json.dumps({"done": True})}
        
        except Exception as err:
            logger.error(f"Streaming error encountered: {err}")
            yield {"data": json.dumps({"error": str(err)})}

    return EventSourceResponse(event_generator())

# --- Synchronous Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if agent_app is None:
        raise HTTPException(status_code=503, detail="Agent workflow is not initialized.")

    try:
        thread = db.query(ChatThread).filter(
            ChatThread.id == request.thread_id, 
            ChatThread.user_id == current_user.id
        ).first()

        if not thread:
            thread = ChatThread(id=request.thread_id, user_id=current_user.id, title="Advisory Session")
            db.add(thread)
            db.commit()

        langchain_messages = []
        for msg in request.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        profile = UserProfile(**request.user_profile) if request.user_profile else UserProfile()
        initial_state = {
            "messages": langchain_messages,
            "user_profile": profile,
            "retrieved_policies": [],
            "next_action": ""
        }

        config = {"configurable": {"thread_id": request.thread_id}}
        final_state = await agent_app.ainvoke(initial_state, config=config)

        ai_reply = final_state["messages"][-1].content
        updated_profile = final_state["user_profile"].model_dump()

        return ChatResponse(reply=ai_reply, user_profile=updated_profile)
    except Exception as e:
        logger.error(f"Chat API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Document Ingestion Endpoint ---
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

class PDFExportRequest(BaseModel):
    user_profile: Dict[str, Any]
    advisory_text: str

@app.post("/export_advisory_pdf")
def export_advisory_pdf_endpoint(
    req: PDFExportRequest,
    current_user: User = Depends(get_current_user)
):
    """Generates and streams a downloadable PDF policy advisory brief."""
    try:
        pdf_bytes = generate_pdf_advisory_report(
            user_profile=req.user_profile,
            advisory_text=req.advisory_text
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=term_life_advisory_brief.pdf"}
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)