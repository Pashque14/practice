import streamlit as st
import os
import warnings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

CHROMA_PATH = "chroma_db"
DOCS_DIR = "docs"

# 1. Настройки страницы (Обязательно первой строкой!)
st.set_page_config(page_title="RAG База Знаний", page_icon="📚", layout="wide")


# 2. Кэширование тяжелых моделей
# Это важнейшая функция! Она загружает LLM и базы только 1 раз при запуске,
# иначе интерфейс зависал бы на 5 секунд после каждого клика.
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = OllamaLLM(model="saiga")
    reranker = CrossEncoder("DiTy/cross-encoder-russian-msmarco")
    return db, llm, reranker


def get_available_documents():
    if not os.path.exists(DOCS_DIR):
        return []
    return [f for f in os.listdir(DOCS_DIR) if f.endswith('.pdf')]


# --- ИНТЕРФЕЙС ---

st.title("📚 Корпоративная база знаний (RAG)")
st.markdown("Задайте вопрос, и нейросеть найдет ответ в загруженных документах.")

# Загрузка системы (покажет крутилку при первом запуске)
with st.spinner("Загрузка нейросетей и базы данных..."):
    db, llm, reranker = load_system()

# --- БОКОВАЯ ПАНЕЛЬ (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Настройки поиска")
    docs_list = get_available_documents()

    # Выпадающий список документов
    options = ["Искать везде"] + docs_list
    selected_doc = st.selectbox("Где искать ответ?", options)

    filter_path = None
    if selected_doc != "Искать везде":
        filter_path = os.path.join(DOCS_DIR, selected_doc)
        st.success(f"Фильтр включен: {selected_doc}")
    else:
        st.info("Поиск по всей базе")

# --- ИСТОРИЯ ЧАТА ---
# Сохраняем сообщения в сессии, чтобы они не пропадали
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовываем старые сообщения
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Посмотреть источники"):
                st.markdown(msg["sources"])

# --- ПОЛЕ ВВОДА ---
if query := st.chat_input("Спросите что-нибудь"):

    # 1. Выводим вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Выводим ответ ассистента
    with st.chat_message("assistant"):
        with st.spinner("Анализирую документы и генерирую ответ..."):

            # --- ЛОГИКА ПОИСКА (Как в консоли) ---
            search_kwargs = {"k": 15}
            if filter_path:
                search_kwargs["filter"] = {"source": filter_path}

            base_retriever = db.as_retriever(search_kwargs=search_kwargs)
            raw_docs = base_retriever.invoke(query)

            if not raw_docs:
                st.warning("В выбранном источнике нет релевантной информации.")
            else:
                # Реранжирование
                sentence_pairs = [[query, doc.page_content] for doc in raw_docs]
                scores = reranker.predict(sentence_pairs)

                scored_docs = list(zip(raw_docs, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                final_docs = [doc for doc, score in scored_docs[:3]]

                context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])

                prompt = f"""Ты — точный ИИ-аналитик. Твоя задача — извлечь ответ на вопрос пользователя из предоставленного текста.
Строгие правила:
1. Используй ТОЛЬКО факты из текста ниже.
2. Внимательно следи за логикой текста: не приписывай действия одних субъектов другим.
3. Если в тексте нет однозначного ответа на вопрос, выведи только одну фразу: "Информация не найдена".

Текст:
{context_text}

Вопрос: {query}
Ответ:"""

                # Получаем ответ от Saiga
                answer = llm.invoke(prompt).strip()

                # Печатаем ответ в интерфейс
                st.markdown(answer)

                # Формируем красивый текст источников
                sources_md = ""
                for i, (doc, score) in enumerate(scored_docs[:3], 1):
                    source = doc.metadata.get('source', 'Неизвестный файл')
                    page = doc.metadata.get('page', '?')
                    sources_md += f"**{i}. {os.path.basename(source)}** (Стр. {page}) — *Релевантность: {score:.2f}*\n\n"

                # Прячем источники в выпадающую плашку (спойлер)
                with st.expander("Посмотреть источники"):
                    st.markdown(sources_md)

                # Сохраняем в историю
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_md
                })