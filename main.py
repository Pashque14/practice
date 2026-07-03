import os
import warnings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Напрямую импортируем оригинальный класс реранкера
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

CHROMA_PATH = "chroma_db"
DOCS_DIR = "docs"


def get_available_documents():
    if not os.path.exists(DOCS_DIR):
        return []
    # Добавили новые расширения!
    allowed_extensions = ('.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg')
    return [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(allowed_extensions)]


def main():
    print("1. Подключаем базовые эмбеддинги и БД...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print("2. Подключаем локальную LLM...")
    llm = OllamaLLM(model="saiga")

    print("3. Подключаем Cross-Encoder (Реранкер)...")

    # Используем надежную русскую модель от DiTy
    reranker = CrossEncoder("DiTy/cross-encoder-russian-msmarco")

    all_docs_list = get_available_documents()

    print("\n" + "=" * 60)
    print("✅ СИСТЕМА С РУЧНЫМ РЕРАНКЕРОМ ГОТОВА К РАБОТЕ!")
    print("=" * 60)

    while True:
        query = input("\nВаш вопрос (или 'выход'): ").strip()
        if query.lower() in ['выход', 'exit', 'quit']:
            break
        if not query:
            continue

        print("🔍 Выполняю предварительный анализ базы (Pre-search)...")

        # ЭТАП 1: ПРИСТРЕЛОЧНЫЙ ПОИСК
        raw_results = db.similarity_search(query, k=50)

        suggested_files = []
        for doc in raw_results:
            source_path = doc.metadata.get('source', '')
            filename = os.path.basename(source_path)
            if filename and filename not in suggested_files:
                suggested_files.append(filename)

        top_3_suggested = suggested_files[:3]

        print("\nГде будем искать ответ?")
        print("0. Искать везде (По всей базе)")
        for i, fname in enumerate(top_3_suggested, 1):
            print(f"{i}. [Рекомендация] Искать только в файле: {fname}")
        print("4. Выбрать вручную из полного списка документов")

        choice = input("\nВаш выбор (цифра): ").strip()
        if choice.lower() in ['выход', 'exit', 'quit']:
            break

        try:
            choice_idx = int(choice)
        except ValueError:
            print("❌ Ошибка: нужно ввести цифру.")
            continue

        filter_path = None

        if choice_idx == 0:
            print("-> Режим: Поиск по всей базе")
        elif 1 <= choice_idx <= len(top_3_suggested):
            selected_file = top_3_suggested[choice_idx - 1]
            filter_path = os.path.join(DOCS_DIR, selected_file)
            print(f"-> Режим: Поиск только в '{selected_file}'")
        elif choice_idx == 4:
            print("\nПолный список документов в базе:")
            for i, doc_name in enumerate(all_docs_list, 1):
                print(f"{i}. {doc_name}")
            sub_choice = input("Выберите номер документа: ").strip()
            try:
                sub_idx = int(sub_choice)
                if 1 <= sub_idx <= len(all_docs_list):
                    selected_file = all_docs_list[sub_idx - 1]
                    filter_path = os.path.join(DOCS_DIR, selected_file)
                    print(f"-> Режим: Поиск только в '{selected_file}'")
                else:
                    print("❌ Неверный номер. Отмена запроса.")
                    continue
            except ValueError:
                print("❌ Ошибка ввода. Отмена запроса.")
                continue
        else:
            print("❌ Неверный выбор. Отмена запроса.")
            continue

        # ==========================================
        # ЭТАП 2: ФИНАЛЬНЫЙ ПОИСК + РЕРАНЖИРОВАНИЕ
        # ==========================================
        print("⚙️ Достаю фрагменты из базы и переранжирую их (Reranking)...")

        search_kwargs = {"k": 15}
        if filter_path:
            just_filename = os.path.basename(filter_path)

            # Обходим баги Windows: ищем точную строку в памяти самой базы
            exact_source = None
            all_data = db.get()
            for meta in all_data['metadatas']:
                if meta and 'source' in meta and just_filename in meta['source']:
                    exact_source = meta['source']
                    break

            # Если нашли точный путь — используем самый надежный фильтр (строгое равенство)
            if exact_source:
                search_kwargs["filter"] = {"source": exact_source}
            else:
                print(f"\n⚠️ Текст из файла {just_filename} не найден в базе!")
                continue

        base_retriever = db.as_retriever(search_kwargs=search_kwargs)

        # 1. Получаем 15 сырых фрагментов из базы
        raw_docs = base_retriever.invoke(query)

        if not raw_docs:
            print("\n=== ОТВЕТ ===")
            print("В выбранном источнике нет релевантной информации.")
            continue

        # 2. Формируем пары (Вопрос, Текст_фрагмента) для оценки реранкером
        sentence_pairs = [[query, doc.page_content] for doc in raw_docs]

        # 3. Реранкер оценивает каждую пару от 0 до 1 (насколько текст отвечает на вопрос)
        scores = reranker.predict(sentence_pairs)

        scored_docs = list(zip(raw_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # --- НОВАЯ ЗАЩИТА: Отсекаем мусор ---
        # Оставляем только те документы, где реранкер уверен хотя бы на 15%
        good_docs = [(doc, score) for doc, score in scored_docs if score > 0.15]

        if not good_docs:
            print("\n=== ОТВЕТ ===")
            print("Алгоритм не нашел в базе точных совпадений для ответа (всё отсеяно по низкой релевантности).")
            print("\n" + "-" * 60)
            continue

        # Берем ДО 3 хороших кусков (если нашелся только 1 хороший — возьмем только 1)
        final_docs = [doc for doc, score in good_docs[:3]]
        context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])

        # ... дальше идет формирование prompt (без изменений) ...

        # Формируем текст для LLM из лучших кусков
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

        print("🧠 Генерирую итоговый ответ нейросетью...")
        answer = llm.invoke(prompt)

        print("\n=== ОТВЕТ ===")
        print(answer.strip())

        print("\n=== ИСТОЧНИКИ (Топ-3 после переранжирования) ===")
        # Выводим источники, чтобы видеть, какие фрагменты победили
        for i, (doc, score) in enumerate(scored_docs[:3], 1):
            source = doc.metadata.get('source', 'Неизвестный файл')
            page = doc.metadata.get('page', 'Неизвестная страница')
            # Для наглядности выведем оценку реранкера!
            print(f"{i}. Файл: {source} (Страница: {page}) [Оценка релевантности: {score:.2f}]")
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()