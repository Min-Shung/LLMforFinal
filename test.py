import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma.vectorstores import Chroma

# 初始化語音識別
recognizer = sr.Recognizer()

def initialize_text_processing():
    # 載入和分割文本
    file_path = 'commodity.txt'
    loader = TextLoader(file_path=file_path, encoding='utf-8')
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10, separators=["\n\n", "\n", " ", ".", ","]
    )
    chunks = text_splitter.split_documents(pages)
    embeddings_model = OllamaEmbeddings(model="llama3.2")
    vectors = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
    db_directory = 'db'
    if not os.path.exists(db_directory):
        db = Chroma.from_documents(
            collection_name='product_recommendation', documents=chunks,
            embedding=embeddings_model, persist_directory=db_directory
        )
    else:
        db = Chroma(
            collection_name='product_recommendation',
            persist_directory=db_directory,
            embedding_function=embeddings_model
        )
    return db, embeddings_model

def recommend_product_multiround(db, embeddings_model, chat_history, query):
    # 搜索相關的產品資訊
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    
    # 加入最新問題到歷史對話
    chat_history.append({"role": "user", "content": f"根據以下產品資訊推薦產品:\n{context}\n問題: {query}"})
    
    # 構建 Prompt
    prompt = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])
    
    # 使用模型生成回應
    try:
        # 假設 chat_model 是一個可用的 Chat 模型
        chat_model = ChatOllama(model="llama3.2")
        response = chat_model.invoke(prompt)
        
        # 將模型的回應加入對話歷史
        chat_history.append({"role": "assistant", "content": response.content})
        return response.content
    except Exception as e:
        st.error(f"獲取推薦時發生錯誤: {e}")
        return "抱歉，無法生成推薦結果。"

def handle_voice_input():
    # 提供語音提示
    prompt_text = "請開始說話..."
    audio_file = text_to_speech(prompt_text)
    
    if audio_file:
        st.audio(audio_file, format="audio/mp3")
    
    with sr.Microphone() as source:
        st.info("正在聽取語音...")
        try:
            audio = recognizer.listen(source, timeout=10)
            recognized_text = recognizer.recognize_google(audio, language="zh-TW")
            st.success(f"語音識別成功: {recognized_text}")
            return recognized_text
        except sr.UnknownValueError:
            st.error("無法識別語音，請再試一次。")
        except sr.RequestError:
            st.error("語音服務出現問題，請檢查網路連線。")
        except Exception as e:
            st.error(f"發生錯誤: {e}")
    return ""

def text_to_speech(response_text):
    try:
        # 使用 gTTS 生成語音並設置語言為繁體中文
        tts = gTTS(text=response_text, lang="zh-TW")  # 使用繁體中文
        
        # 儲存語音到臨時檔案
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        
        # 播放音檔
        return temp_audio.name  # 回傳音檔名稱，便於播放
    except Exception as e:
        st.error(f"無法生成語音: {e}")
        return None

def main():
    st.title("推薦購物系統")

    # 初始化
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if "response_text" not in st.session_state:
        st.session_state["response_text"] = ""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "system", "content": "你是一個協助別人尋找想要的用品的機器人，請用繁體中文回答問題，如果有輸入價格期許，請回答符合價格需求的商品，商品必須是實際存在且可行的，你必須回答1到3個不同類型、面向、用途的商品，並總結大概需要花費的台幣"}
        ]

    db, embeddings_model = initialize_text_processing()

    # 使用者輸入
    user_input = st.text_area("請輸入需求", value=st.session_state["user_input"], key="text_input")
    if st.button("語音輸入"):
        st.session_state["user_input"] = handle_voice_input()

    if st.button("送出"):
        query = st.session_state["user_input"].strip()
        if query:
            response = recommend_product_multiround(db, embeddings_model, st.session_state["chat_history"], query)
            st.session_state["response_text"] = response
            st.text("推薦商品:")
            st.write(response)

            # 生成和播放語音
            audio_file = text_to_speech(st.session_state["response_text"])
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

            # 清除輸入
            st.session_state["user_input"] = ""
        else:
            st.warning("請輸入需求!")

if __name__ == "__main__":
    main()
