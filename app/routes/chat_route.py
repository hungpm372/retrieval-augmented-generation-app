from fastapi import APIRouter

from app.rag.chain import conversational_rag_chain
from app.schemas.chat_schema import ChatOutputSchema, ChatInputSchema

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("", response_model=ChatOutputSchema)
async def chat(request_body: ChatInputSchema):
    response = conversational_rag_chain.invoke({"input": request_body.input}, config={
        "configurable": {"session_id": "1"}
    })
    return ChatOutputSchema(answer=response["answer"])
