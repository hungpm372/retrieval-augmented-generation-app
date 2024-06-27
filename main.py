import os
import bs4

from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=".env.dev")

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

loader = WebBaseLoader(
    web_paths=("https://znews.vn/doi-bong-gay-soc-nhat-tai-vck-euro-2024-post1483303.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_="the-article-body",
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

result = rag_chain.invoke("Đội nào thua?")
print(result)
