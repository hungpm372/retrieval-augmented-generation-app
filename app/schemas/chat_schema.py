from pydantic import BaseModel


class BaseChatSchema(BaseModel):
    pass


class ChatInputSchema(BaseChatSchema):
    input: str


class ChatOutputSchema(BaseChatSchema):
    answer: str
