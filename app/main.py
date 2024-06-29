from fastapi import FastAPI
from langserve import add_routes

from app.rag.chain import conversational_rag_chain
from app.routes import chat_router

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="RAG conversational chain with LangChain",
)

add_routes(
    app,
    conversational_rag_chain,
    path="/openai",
)

app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
