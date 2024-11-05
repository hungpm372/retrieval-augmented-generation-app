from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from app.rag.llm import llm
from app.rag.vectorstore import retriever

system_prompt = (
    "Bạn là trợ lý hỗ trợ đặt tour du lịch. "
    "Sử dụng các thông tin có sẵn sau đây để trả lời câu hỏi "
    "một cách chính xác. Nếu không biết câu trả lời, hãy nói rằng "
    "Tôi không biết. Trả lời tối đa ba câu và giữ cho câu trả lời ngắn gọn."
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Bạn là trợ lý hỗ trợ đặt tour du lịch. "
    "Sử dụng các thông tin có sẵn sau đây để trả lời câu hỏi "
    "một cách chính xác. Nếu không biết câu trả lời, hãy nói rằng "
    "Tôi không biết. Trả lời tối đa ba câu và giữ cho câu trả lời ngắn gọn."
    "answer concise."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
