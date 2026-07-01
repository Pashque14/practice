import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "docs/"
# Папка, куда сохранится наша готовая векторная база
CHROMA_PATH = "chroma_db"


def load_and_split_documents():
    print("1. Загрузка документов из папки 'docs/'...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"-> Текст разбит на {len(chunks)} фрагментов.\n")
    return chunks


def create_vector_db(chunks):
    print("2. Загружаем модель эмбеддингов (потребуется скачать около 450 МБ при первом запуске)...")
    # Используем легковесную модель, которая отлично понимает русский язык
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("3. Создаем векторную базу данных ChromaDB и сохраняем на диск...")
    # Создаем базу и указываем директорию для физического сохранения
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"-> Готово! Векторная база данных сохранена в папку: {CHROMA_PATH}")
    return db


if __name__ == "__main__":
    # 1. Загружаем и бьем текст
    my_chunks = load_and_split_documents()

    # 2. Создаем базу, если чанки есть
    if my_chunks:
        create_vector_db(my_chunks)