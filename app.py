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
# Добавили новые расширения для отображения в меню
ALLOWED_EXTENSIONS = ('.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg')

# 1. Настройки страницы (Обязательно первой строкой!)
st.set_page_config(page_title="RAG База Знаний", page_icon="📚", layout="wide")


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
    # Фильтруем по всем поддерживаемым форматам
    return [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(ALLOWED_EXTENSIONS)]


# --- ИНТЕРФЕЙС ---

st.title("📚 Корпоративная база знаний (RAG)")
st.markdown("Задайте вопрос, и нейросеть найдет ответ в загруженных документах.")

with st.spinner("Загрузка нейросетей и базы данных..."):
    db, llm, reranker = load_system()

# --- БОКОВАЯ ПАНЕЛЬ (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Настройки поиска")
    docs_list = get_available_documents()

    options = ["Искать везде"] + docs_list
    selected_doc = st.selectbox("Где искать ответ?", options)

    filter_filename = None
    if selected_doc != "Искать везде":
        filter_filename = selected_doc
        st.success(f"Фильтр включен: {selected_doc}")
    else:
        st.info("Поиск по всей базе")

# --- ИСТОРИЯ ЧАТА ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Посмотреть источники"):
                st.markdown(msg["sources"])

# --- ПОЛЕ ВВОДА ---
if query := st.chat_input("Спросите что-нибудь"):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую документы и генерирую ответ..."):

            search_kwargs = {"k": 50}

            # --- УМНЫЙ ФИЛЬТР ПУТЕЙ WINDOWS (Перенесли из main.py) ---
            if filter_filename:
                exact_source = None
                all_data = db.get()
                if all_data and 'metadatas' in all_data:
                    for meta in all_data['metadatas']:
                        if meta and 'source' in meta and filter_filename in meta['source']:
                            exact_source = meta['source']
                            break

                if exact_source:
                    search_kwargs["filter"] = {"source": exact_source}
                else:
                    st.warning(f"Текст из файла {filter_filename} не найден в базе!")
                    # Ставим заглушку, чтобы поиск ничего не нашел, раз файла нет в базе
                    search_kwargs["filter"] = {"source": "NON_EXISTENT_FILE"}

            base_retriever = db.as_retriever(search_kwargs=search_kwargs)
            raw_docs = base_retriever.invoke(query)

            if not raw_docs:
                st.warning("В выбранном источнике нет релевантной информации.")
            else:
                sentence_pairs = [[query, doc.page_content] for doc in raw_docs]
                scores = reranker.predict(sentence_pairs)

                scored_docs = list(zip(raw_docs, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # --- НОВАЯ ЗАЩИТА: Отсекаем мусор ---
                # Оставляем только те документы, где реранкер уверен хотя бы на 15%
                good_docs = [(doc, score) for doc, score in scored_docs if score > 0.15]

                if not good_docs:
                    st.warning("Алгоритм не нашел в базе точных совпадений для ответа.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "К сожалению, в загруженных документах нет информации для ответа на этот вопрос."
                    })
                    st.rerun()

                # Берем ДО 3 хороших кусков (если нашелся только 1 хороший — возьмем только 1)
                final_docs = [doc for doc, score in good_docs[:3]]
                context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])

                # ... дальше идет формирование prompt (без изменений) ...

                prompt = f"""Ты — точный ИИ-аналитик. Твоя задача — извлечь ответ на вопрос пользователя из предоставленного текста.
Строгие правила:
1. Используй ТОЛЬКО факты из текста ниже.
2. Внимательно следи за логикой текста: не приписывай действия одних субъектов другим.
3. Если в тексте нет однозначного ответа на вопрос, выведи только одну фразу: "Информация не найдена".

Текст:
{context_text}

Вопрос: {query}
Ответ:"""

                answer = llm.invoke(prompt).strip()
                st.markdown(answer)

                sources_md = ""
                for i, (doc, score) in enumerate(scored_docs[:3], 1):
                    source = doc.metadata.get('source', 'Неизвестный файл')
                    page = doc.metadata.get('page', '?')
                    sources_md += f"**{i}. {os.path.basename(source)}** (Стр. {page}) — *Релевантность: {score:.2f}*\n\n"

                with st.expander("Посмотреть источники"):
                    st.markdown(sources_md)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_md
                })