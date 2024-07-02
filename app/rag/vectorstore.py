from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGEmbedding
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core import get_settings
from app.rag.loader import loader

settings = get_settings()

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

connection_string = settings.DATABASE_URI

collection_name = "state_of_the_union"

# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(),
    connection=connection_string,
    collection_name=collection_name,
    use_jsonb=True,
    pre_delete_collection=False,
)

vectorstore.add_documents(splits)

retriever = vectorstore.as_retriever()
