import streamlit as st
import ollama

def main():
    st.title("推薦購物")

    #用戶輸入框
    user_input = st.text_area("您有什麼需求","")

    #按下按鈕
    if st.button("送出"):
        if user_input:
            #use ollama to get recommendation
            response = ollama.chat(model = 'llama3', messages = [{'role': 'user', 'content': user_input}])

            #show response
            st.text("推薦商品:")
            st.write(response['message']['content'])
        else:
            st.warning("請輸入需求!")

if __name__ == "__main__":
    main()