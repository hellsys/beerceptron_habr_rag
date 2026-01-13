import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000.")


st.set_page_config(page_title="HABR RAG", layout="wide")
st.title("🔍 HABR RAG Chat")

response = requests.get(API_URL+"/health")

print("Health check response:", response.json())

# Инициализируем историю
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем прошлые сообщения
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Ввод
user_query = st.chat_input("Введите вопрос:")

if user_query:
    # Добавляем в историю
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Отправляем запрос
    with st.spinner("Генерация ответа..."):
        response = requests.post(API_URL+"api/v1/rag", json={"query": st.session_state.messages})
        answer = response.json().get("answer", "Ошибка соединения")

    # Добавляем ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        
        st.write(answer)