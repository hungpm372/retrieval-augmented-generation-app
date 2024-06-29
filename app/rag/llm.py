import os

from langchain_openai import ChatOpenAI

from app.core import get_settings

settings = get_settings()

llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY)
