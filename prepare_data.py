import os
import tkinter as tk
from tkinter import filedialog
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# ВАЖНО: Укажи здесь путь к установленному Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CHROMA_PATH = "chroma_db"
ALLOWED_EXTENSIONS = ('.pdf', '.txt', '.docx', '.doc', '.png', '.jpg', '.jpeg')


def select_target():
    """Открывает нативные окна Windows для выбора файлов или папок"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("\n" + "=" * 50)
    print("📂 ИНТЕРФЕЙС ЗАГРУЗКИ ДОКУМЕНТОВ В БАЗУ ЗНАНИЙ")
    print("=" * 50)
    print("Что вы хотите загрузить?")
    print("1. Один файл")
    print("2. Группу файлов")
    print("3. Папку целиком (включая все поддерживаемые форматы внутри)")

    choice = input("\nВаш выбор (1/2/3): ").strip()

    file_types = [
        ("Поддерживаемые документы", "*.pdf *.txt *.doc *.docx *.png *.jpg *.jpeg"),
        ("Все файлы", "*.*")
    ]

    if choice == '1':
        print("Открываю диалог выбора файла...")
        file_path = filedialog.askopenfilename(title="Выберите документ для БД", filetypes=file_types)
        if file_path:
            return [file_path]

    elif choice == '2':
        print("Открываю диалог выбора группы файлов...")
        file_paths = filedialog.askopenfilenames(title="Выберите документы (держите Ctrl)", filetypes=file_types)
        if file_paths:
            return list(file_paths)

    elif choice == '3':
        print("Открываю диалог выбора папки...")
        dir_path = filedialog.askdirectory(title="Выберите папку с документами")
        if dir_path:
            found_files = [
                os.path.join(dir_path, f) for f in os.listdir(dir_path)
                if f.lower().endswith(ALLOWED_EXTENSIONS)
            ]
            print(f"Найдено поддерживаемых файлов внутри: {len(found_files)}")
            return found_files

    else:
        print("❌ Неверный выбор.")

    return []


def load_document(file_path):
    """Определяет тип файла и использует нужный парсер, включая OCR для PDF-сканов"""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    try:
        if ext == '.pdf':
            # Сначала пробуем быстрый метод (обычный текстовый слой)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Считаем, сколько всего текста удалось вытащить
            total_text = "".join([d.page_content for d in docs]).strip()
            # Если текста почти нет — перед нами скан или картинка!

            if len(total_text) < 50:
                print(f"⚠️ Текст в {filename} не найден. Запускаю OCR (распознавание скана, это займет время)...")
                ocr_docs = []
                pdf_document = fitz.open(file_path)
                # Проходимся по каждой странице документа
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    # Превращаем страницу в картинку (dpi=200 оптимально для распознавания)
                    pix = page.get_pixmap(dpi=200)
                    # Конвертируем внутренний формат PyMuPDF в формат библиотеки Pillow (Image)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # Натравливаем Tesseract на картинку
                    text = pytesseract.image_to_string(img, lang='rus+eng')

                    if text.strip():
                        ocr_docs.append(Document(
                            page_content=text,
                            metadata={"source": file_path, "page": page_num + 1}
                        ))
                print(f"✅ OCR завершен. Распознано страниц: {len(ocr_docs)}")
                return ocr_docs
            # Если текста много, возвращаем результат быстрого метода
            return docs

        elif ext == '.txt':
            # autodetect_encoding спасет от падений на кодировках вроде cp1251
            loader = TextLoader(file_path, autodetect_encoding=True)
            return loader.load()

        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
            return loader.load()

        elif ext in ['.png', '.jpg', '.jpeg']:
            img = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(img, lang='rus+eng')

            if not extracted_text.strip():
                print(f"⚠️ Текст на изображении '{filename}' не найден.")
                return []
            doc = Document(
                page_content=extracted_text,
                metadata={"source": file_path, "page": 1}
            )
            return [doc]

        else:
            return []
    except Exception as e:
        print(f"❌ Ошибка при чтении {filename}: {e}")
        return []


def main():
    # 1. Запрашиваем у пользователя пути к файлам через GUI Windows
    files_to_process = select_target()

    if not files_to_process:
        print("Отмена. Файлы не выбраны.")
        return

    all_documents = []

    # 2. Загружаем выбранные файлы
    print("\nЧтение и парсинг файлов...")
    for file_path in files_to_process:
        docs = load_document(file_path)
        if docs:
            all_documents.extend(docs)
            print(f"✅ Успешно загружен: {os.path.basename(file_path)}")

    if not all_documents:
        print("❌ Не удалось извлечь текст из выбранных файлов.")
        return

    # 3. Нарезка текста
    print("\nНарезка текста на фрагменты (чанки)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    print(f"Получено {len(chunks)} фрагментов.")

    # 4. Векторизация и добавление в базу данных
    print("Создание эмбеддингов и сохранение в базу данных...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Chroma автоматически добавит новые данные в существующую папку chroma_db
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    print(f"\n🎉 Файлы успешно добавлены в базу! (Путь: {CHROMA_PATH})")


if __name__ == "__main__":
    main()