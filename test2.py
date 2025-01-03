import streamlit as st
import ollama
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 設置 OpenAI API 金鑰
embeddings_model = OpenAIEmbeddings(openai_api_key='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')

# loader = PyPDFLoader(file_path='./rag1.pdf',extract_images=False) # file_path=url
loader = TextLoader(file_path='commodity.txt',encoding="utf8") # utf-8 encoding

pages = loader.load()


if os.path.exists('db'):
    # Step 4
    db = Chroma(collection_name='commodity',persist_directory='db',embedding_function=embeddings_model)
else:
    # Step 2
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=10)                                       
    chunks = text_splitter.split_documents(pages)
    # Step 3
    collection_name = 'commodity'
    db = Chroma.from_documents(collection_name=collection_name,documents=chunks,embedding=embeddings_model,persist_directory='db',collection_metadata={"hnsw:space": "cosine"})
    # Step 4
    # db = Chroma(collection_name=collection_name,persist_directory='./db',embedding_function=embeddings_model)



def main():
    st.title("推薦購物")

    # 初始化語音識別
    recognizer = sr.Recognizer()

    # 初始化輸入框的狀態
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if "response_text" not in st.session_state:
        st.session_state["response_text"] = ""

    #用戶輸入框
    user_input = st.text_area("您有什麼需求", st.session_state["user_input"])

    # 語音輸入按鈕
    if st.button("語音輸入"):
        with sr.Microphone() as source:
            st.info("請開始說話...")
            try:
                audio = recognizer.listen(source, timeout=5)  # 等待用戶輸入語音
                recognized_text = recognizer.recognize_google(audio, language="zh-TW")                
                st.success(f"語音識別成功: {recognized_text}")
                st.session_state["user_input"] = recognized_text  # 更新輸入框內容
            except sr.UnknownValueError:
                st.error("無法識別語音，請再試一次。")
            except sr.RequestError:
                st.error("語音服務出現問題，請檢查網路連線。")
            except Exception as e:
                st.error(f"發生錯誤: {e}")



    if st.button("送出"):
        if user_input:
            #use ollama to get recommendation
            response = ollama.chat(
                model = 'llama2', 
                messages=[
                    {"role": "system", "content": "你是一個商品推薦助手，幫助用戶找到符合需求的商品，並計算大概需要花多少錢。"},
                    {"role": "user", "content": user_input}
                ]
            )

            #show response
            response_text = response['message']['content']
            st.session_state["response_text"] = response_text
            st.text("推薦商品:")
            st.write(response_text)

        else:
            st.warning("請輸入需求!")

    

            # 將回答轉換為語音
            tts = gTTS(text=response_text, lang="zh-TW")
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio.name)

            # 播放音檔
            st.audio(temp_audio.name, format="audio/mp3")
            # 清理臨時文件（若需要）
            # os.unlink(temp_audio.name)

        

if __name__ == "__main__":
    main()