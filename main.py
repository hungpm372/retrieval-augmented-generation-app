from app.rag.chain import conversational_rag_chain


def chat_with_rag():
    print("Chào mừng đến với hệ thống trò chuyện RAG! Nhập 'exit' để thoát.")

    while True:
        user_input = input("Bạn: ")

        # Kiểm tra nếu người dùng muốn thoát
        if user_input.lower() == 'exit':
            print("Cảm ơn bạn đã trò chuyện. Tạm biệt!")
            break

        # Gọi hàm hoặc đối tượng từ chuỗi RAG để lấy phản hồi
        response = conversational_rag_chain.invoke({"input": user_input}, config={
            "configurable": {"session_id": "1"}
        })

        # In phản hồi từ hệ thống
        print("Hệ thống:", response["answer"])


# Gọi hàm để bắt đầu trò chuyện
chat_with_rag()
