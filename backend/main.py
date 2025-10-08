# backend/main.py
import os, io, base64, json, uuid, datetime
from typing import List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
from openai import OpenAI
from sqlalchemy.orm import Session as DBSession


# --- Load Config.json and set globals ---
CONFIG_PATH = "../config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("Missing config.json file")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# -------- Config --------
PINECONE_API_KEY = config["PINECONE_API_KEY"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]
INDEX_NAME = config["INDEX_NAME"]
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
DIMENSION = config["DIMENSION"]
MODEL = config["MODEL"]
# ------------------------

# =========================
# Config & globals
# =========================
EMBED_DIM = DIMENSION
TOP_K = 2
EMBED_SCORE_THRESHOLD = config["EMBED_SCORE_THRESHOLD"]

UPLOAD_DIR = "user_uploads"
LOG_FILE = "judge_logs.jsonl"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# Pinecone + Model
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"🔥 Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy()[0].tolist()

# =========================
# OpenAI (Judge)
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

def judge_match(user_bytes: bytes, candidate_meta: dict) -> (bool, str):
    """Ask LLM if user image matches candidate. Returns (verdict, raw_reply)."""
    b64_user = base64.b64encode(user_bytes).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": "Does the uploaded photo show the same object as the reference? Answer yes or no and explain briefly.",
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_user}"}},
    ]

    ref_url = (candidate_meta or {}).get("image_url", "")
    if isinstance(ref_url, str) and ref_url.startswith("http"):
        content.append({"type": "image_url", "image_url": {"url": ref_url}})
    elif (candidate_meta or {}).get("description"):
        content.append({"type": "text", "text": f"Reference description: {candidate_meta['description']}"})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
    )

    reply = resp.choices[0].message.content.strip()
    verdict = reply.lower().startswith("yes")
    return verdict, reply

# =========================
# DB (SQLite via SQLAlchemy)
# =========================
from sqlalchemy import (
    create_engine, Column, String, DateTime, Boolean, Integer, Float, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

engine = create_engine("sqlite:///scavenger.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed = Column(Boolean, default=False)
    found_items = relationship("FoundItem", back_populates="session", cascade="all, delete-orphan")

class FoundItem(Base):
    __tablename__ = "found_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    item_id = Column(String)         # Pinecone vector id
    name = Column(String)            # metadata name
    user_image_path = Column(String) # saved local path
    pinecone_score = Column(Float)
    llm_verdict = Column(Boolean)
    llm_reply = Column(Text)
    matched_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("Session", back_populates="found_items")
    __table_args__ = (UniqueConstraint("session_id", "item_id", name="unique_session_item"),)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# FastAPI app
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# serve uploaded images back to the client
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# =========================
# Helpers
# =========================
def get_all_items_from_pinecone(max_items: int = 500):
    # "zero vector" trick to pull a page of items + metadata
    res = index.query(vector=[0.0]*EMBED_DIM, top_k=max_items, include_metadata=True)
    items = []
    for m in res.matches or []:
        md = m.metadata or {}
        items.append({
            "id": m.id,
            "name": md.get("name"),
            "description": md.get("description", ""),
            "image_url": md.get("image_url"),
        })
    return items

def count_total_items() -> int:
    return len(get_all_items_from_pinecone())

# =========================
# Endpoints
# =========================
@app.post("/session/start")
def start_session():
    sid = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(Session(session_id=sid))
        db.commit()
    return {"session_id": sid}

@app.get("/session/{session_id}/progress")
def get_progress(session_id: str):
    items = get_all_items_from_pinecone()
    with SessionLocal() as db:
        s = db.get(Session, session_id)
        if not s:
            # If the client presents an unknown session (cleared DB), recreate it
            s = Session(session_id=session_id)
            db.add(s)
            db.commit()
        found = db.query(FoundItem).filter(FoundItem.session_id == session_id).all()
    found_map = {f.item_id: f for f in found}
    # Rehydrate hunt list with found flags + images (served from /uploads/...)
    hydrated = []
    for it in items:
        f = found_map.get(it["id"])
        hydrated.append({
            **it,
            "found": bool(f),
            "user_image_url": f"/uploads/{os.path.basename(f.user_image_path)}" if f and f.user_image_path else None,
            "matched_at": f.matched_at.isoformat() if f else None,
        })
    all_found = all(x["found"] for x in hydrated) if hydrated else False
    return {"items": hydrated, "completed": all_found}

@app.get("/items")
def list_items():
    return {"items": get_all_items_from_pinecone()}

@app.post("/upload")
async def upload_image(
        file: UploadFile = File(...),
        session_id: Optional[str] = Form(None),
        x_session_id: Optional[str] = Header(None),
):
    # Get session id (Form field takes precedence, then Header)
    sid = session_id or x_session_id
    if not sid:
        # If none provided, create one on the fly (client should call /session/start normally)
        sid = str(uuid.uuid4())
        with SessionLocal() as db:
            db.add(Session(session_id=sid))
            db.commit()

    # Ensure session exists
    with SessionLocal() as db:
        s = db.get(Session, sid)
        if not s:
            s = Session(session_id=sid)
            db.add(s)
            db.commit()

    # Save upload locally
    file_bytes = await file.read()
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    stored_name = f"{now}_{short_id}_{safe_name}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    # Embed & Pinecone search
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    vector = embed_image(image)
    result = index.query(vector=vector, top_k=TOP_K, include_metadata=True)

    # Log Pinecone raw matches
    pinecone_log = {
        "timestamp": now,
        "session_id": sid,
        "upload_file": stored_path,
        "pinecone_matches": [
            {"id": m.id, "score": m.score, "metadata": m.metadata} for m in (result.matches or [])
        ],
    }
    with open(LOG_FILE, "a") as logf:
        logf.write(json.dumps({"pinecone_result": pinecone_log}) + "\n")

    # Judge candidates
    for match in (result.matches or []):
        if match.score >= EMBED_SCORE_THRESHOLD:
            verdict, raw_reply = judge_match(file_bytes, match.metadata or {})
            # Log judge decision
            judge_log = {
                "timestamp": now,
                "session_id": sid,
                "upload_file": stored_path,
                "candidate_id": match.id,
                "candidate_meta": match.metadata,
                "pinecone_score": match.score,
                "llm_reply": raw_reply,
                "verdict": verdict,
            }
            with open(LOG_FILE, "a") as logf:
                logf.write(json.dumps({"judge_result": judge_log}) + "\n")

            if verdict:
                # Persist found item if not already in DB
                with SessionLocal() as db:
                    already = db.query(FoundItem).filter(
                        FoundItem.session_id == sid,
                        FoundItem.item_id == match.id
                    ).first()
                    if not already:
                        db.add(FoundItem(
                            session_id=sid,
                            item_id=match.id,
                            name=(match.metadata or {}).get("name"),
                            user_image_path=stored_path,
                            pinecone_score=match.score,
                            llm_verdict=True,
                            llm_reply=raw_reply
                        ))
                        db.commit()

                    # Check completion
                    total = count_total_items()
                    found_count = db.query(FoundItem).filter(FoundItem.session_id == sid).count()
                    is_complete = (found_count >= total and total > 0)
                    if is_complete:
                        sess = db.get(Session, sid)
                        if sess and not sess.completed:
                            sess.completed = True
                            db.commit()

                return {
                    "success": True,
                    "session_id": sid,
                    "item": {
                        "id": match.id,
                        "name": (match.metadata or {}).get("name"),
                        "description": (match.metadata or {}).get("description"),
                        "image_url": (match.metadata or {}).get("image_url"),
                    },
                    "user_image_url": f"/uploads/{stored_name}",
                    "completed": is_complete if 'is_complete' in locals() else False
                }

    return {"success": False, "session_id": sid}

@app.get("/admin/winners")
def list_winners(db: DBSession = Depends(get_db)):
    sessions = db.query(Session).filter(Session.completed == True).all()
    winners = []
    for s in sessions:
        items = [
            {
                "item_id": f.item_id,
                "name": f.name,
                "user_image_url": f"/uploads/{os.path.basename(f.user_image_path)}",
                "pinecone_score": f.pinecone_score,
                "llm_reply": f.llm_reply,
            }
            for f in s.found_items
        ]
        winners.append({
            "session_id": s.session_id,
            "completed_at": s.found_items[-1].matched_at.isoformat() if s.found_items else None,
            "items": items,
        })
    return {"winners": winners}

# Dev runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
