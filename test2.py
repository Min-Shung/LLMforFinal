import streamlit as st
import ollama
import speech_recognition as sr
from gtts import gTTS
import tempfile

def main():
    st.title("推薦購物")

    # Initialize speech recognizer
    recognizer = sr.Recognizer()

    # Initialize input and response states
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if "response_text" not in st.session_state:
        st.session_state["response_text"] = ""

    # User input text area
    user_input = st.text_area("您有什麼需求", st.session_state["user_input"])

    # Voice input button
    if st.button("語音輸入"):
        with sr.Microphone() as source:
            st.info("請開始說話...")
            try:
                audio = recognizer.listen(source, timeout=5)
                recognized_text = recognizer.recognize_google(audio, language="zh-TW")
                st.success(f"語音識別成功: {recognized_text}")
                st.session_state["user_input"] = recognized_text  # Update input box content
            except sr.UnknownValueError:
                st.error("無法識別語音，請再試一次。")
            except sr.RequestError:
                st.error("語音服務出現問題，請檢查網路連線。")
            except Exception as e:
                st.error(f"發生錯誤: {e}")

    # Submit button
    if st.button("送出"):
        if st.session_state["user_input"]:
            # Use Ollama to generate recommendations
            response = ollama.chat(
                model='llama3',
                messages=[
                    {"role": "system", "content": "你是一個商品推薦助手，幫助用戶找到符合需求的商品，並計算大概需要花多少錢。"},
                    {"role": "user", "content": st.session_state["user_input"]}
                ]
            )
            st.session_state["response_text"] = response['message']['content']
        else:
            st.warning("請輸入需求!")

    # Display recommended products
    if st.session_state["response_text"]:
        st.text("推薦商品:")
        st.write(st.session_state["response_text"])

    # Play response button
    if st.session_state["response_text"] and st.button("播放回答"):
        try:
            tts = gTTS(text=st.session_state["response_text"], lang="zh-TW")
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio.name)
            st.audio(temp_audio.name, format="audio/mp3")
        except Exception as e:
            st.error(f"無法生成語音: {e}")

if __name__ == "__main__":
    main()