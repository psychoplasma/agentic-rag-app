import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils import authenticate_vertex_ai


class QuestionRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_documents: list

class ProcessRepository(BaseModel):
    path: str

# Load environment variables from a file(default: .env)
# load_dotenv("/secrets/.env")
load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
CREDENTIALS_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_URI = os.environ.get("BUCKET_URI")

authenticate_vertex_ai(PROJECT_ID, LOCATION, CREDENTIALS_FILE, BUCKET_URI)

# Import the agent after authentication
from src.agentic_rag import ask_agent, build_agent, process_repository

agent = build_agent()

app = FastAPI()

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    try:
        user_id = "u-abc123"
        thread_id = "abcd1234"
    
        result = ask_agent(agent, request.query, thread_id, user_id)

        return AnswerResponse(
            question=request.query,
            answer=result["answer"],
            source_documents=result["context"],
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process-repo")
async def process_repo(repo: ProcessRepository):
    try:
        num_chunks = process_repository(repo.path)
        return {"message": f"Processed {num_chunks} code files"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
