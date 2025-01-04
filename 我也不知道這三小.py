from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma.vectorstores import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
import os
import streamlit as st
import ollama
import speech_recognition as sr
from gtts import gTTS
import tempfile

# Step 1: 加載文本
file_path = 'commodity.txt'
loader = TextLoader(file_path=file_path, encoding='utf-8')
pages = loader.load()

# Step 2: 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ".", ","]
)
chunks = text_splitter.split_documents(pages)
print(f"分割後的文本段落數量: {len(chunks)}")

# Step 3: 嵌入模型初始化
embeddings_model = OllamaEmbeddings(model="llama3.2")

# 將文本嵌入為向量
documents = [chunk.page_content for chunk in chunks]
vectors = embeddings_model.embed_documents(documents)
print(f"完成嵌入的文本數量: {len(vectors)}")

# Step 4: 建立向量檢索數據庫
db_directory = 'db'
if not os.path.exists(db_directory):
    db = Chroma.from_documents(
        collection_name='product_recommendation',
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=db_directory
    )
else:
    db = Chroma(collection_name='product_recommendation', persist_directory=db_directory, embedding_function=embeddings_model)

# Step 5: 初始化語言模型
chat_model = ChatOllama(model="taide_model", base_url="http://localhost:11434")

# Step 6: 查詢流程
def recommend_product(query):
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    
    prompt = f"""根據以下產品資訊推薦產品:
    {context}
    問題: {query}"""
    
    response_text = chat_model.invoke(prompt).content
    return response_text

# Step 7: 初始化聊天記憶
memory = SQLChatMessageHistory(
    session_id="product_chat",
    connection="sqlite:///chat_memory.db"
)

# Step 8: 模擬對話與記憶
from langchain_core.messages import HumanMessage, AIMessage
def chat_with_memory(user_query):
    # 將用戶輸入轉為 HumanMessage
    memory.add_message(HumanMessage(content=user_query))
    # 執行產品推薦
    response_text = recommend_product(user_query)
    # 將 AI 回覆轉為 AIMessage
    memory.add_message(AIMessage(content=response_text))
    return response_text

# 測試查詢
# query = "推薦一款適合拍照的手機"
# response = chat_with_memory(query)
# print("AI 回覆:", response)

# # 模擬對話
# user_query = "我想學游泳"
# response = chat_with_memory(user_query)
# print("AI 回覆:", response)

def main():
    st.title("推薦購物")

    # 初始化語音識別
    recognizer = sr.Recognizer()

    # 初始化輸入框的狀態
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "response_text" not in st.session_state:
        st.session_state["response_text"] = ""

    #用戶輸入框
    user_query = st.text_area("您有什麼需求", st.session_state["user_query"])

    # 語音輸入按鈕
    if st.button("語音輸入"):
        with sr.Microphone() as source:
            st.info("請開始說話...")
            try:
                audio = recognizer.listen(source, timeout=5)  # 等待用戶輸入語音
                recognized_text = recognizer.recognize_google(audio, language="zh-TW")                
                st.success(f"語音識別成功: {recognized_text}")
                st.session_state["user_query"] = recognized_text  # 更新輸入框內容
            except sr.UnknownValueError:
                st.error("無法識別語音，請再試一次。")
            except sr.RequestError:
                st.error("語音服務出現問題，請檢查網路連線。")
            except Exception as e:
                st.error(f"發生錯誤: {e}")



    if st.button("送出"):
        if user_query:
            #use ollama to get recommendation
            response = ollama.chat(
                model = 'llama3', 
                messages=[
                    {"role": "system", "content": "你是一個商品推薦助手，請使用中文回答，幫助用戶找到符合需求的商品，並計算大概需要花多少錢。"},
                    {"role": "user", "content": user_query}
                ]
            )

            #show response
            response_text = response['message']['content']
            st.session_state["response_text"] = response_text
            st.text("推薦商品:")
            st.write(response_text)

        else:
            st.warning("請輸入需求!")

    # 播放回答的按鈕
    if st.session_state["response_text"]:
        st.text("suggestion:") 
        st.write(st.session_state["response_text"])
        if st.button("播放回答"):
            try:
                # 將回答轉換為語音
                tts = gTTS(text=st.session_state["response_text"], lang="zh-TW")
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(temp_audio.name)

                # 播放音檔
                st.audio(temp_audio.name, format="audio/mp3")
                # 清理臨時文件（若需要）
                # os.unlink(temp_audio.name)
            except Exception as e:
                st.error(f"無法生成語音: {e}")

        

if __name__ == "__main__":
    main()