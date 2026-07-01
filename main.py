from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import warnings

# Отключаем вывод назойливых предупреждений от библиотек
warnings.filterwarnings("ignore")

CHROMA_PATH = "chroma_db"


def main():
    print("1. Подключаем эмбеддинги и базу данных...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("2. Подключаем локальную LLM...")
    llm = OllamaLLM(model="saiga")

    print("\n" + "=" * 50)
    print("✅ СИСТЕМА ГОТОВА! Можно задавать вопросы.")
    print("Для выхода напишите 'выход'.")
    print("=" * 50 + "\n")

    while True:
        query = input("Ваш вопрос: ")
        if query.lower() in ['выход', 'exit', 'quit']:
            print("Завершение работы...")
            break

        if not query.strip():
            continue

        print("Ищу в базе и генерирую ответ (это может занять несколько секунд)...")

        found_docs = retriever.invoke(query)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in found_docs])

        # Новый, более четкий промпт для модели
        prompt = f"""Опираясь ТОЛЬКО на следующий текст, ответь на вопрос. 
Если в тексте нет информации для ответа, напиши ровно одну фразу: "Я не знаю". Не придумывай ничего от себя.

Текст из базы знаний:
{context_text}

Вопрос: {query}
Ответ:"""

        answer = llm.invoke(prompt)

        print("\n=== ОТВЕТ ===")
        print(answer.strip())

        print("\n=== ИСТОЧНИКИ ===")
        for i, doc in enumerate(found_docs, 1):
            source = doc.metadata.get('source', 'Неизвестный файл')
            page = doc.metadata.get('page', 'Неизвестная страница')
            print(f"{i}. Файл: {source} (Страница: {page})")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()