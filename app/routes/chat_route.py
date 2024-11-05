from fastapi import APIRouter

from app.rag.chain import conversational_rag_chain

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("")
async def chat(request_body):
    response = conversational_rag_chain.invoke({"input": request_body.input}, config={
        "configurable": {"session_id": "1"}
    })
    return {
        'answer': response["answer"]
    }
