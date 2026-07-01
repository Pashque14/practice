from langchain_community.document_loaders import PyPDFDirectoryLoader

print("Сканируем папку docs/...")
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()

# Используем множество (set), чтобы оставить только уникальные названия файлов.
# Мы проходимся по всем загруженным страницам и вытаскиваем поле 'source' из метаданных.
loaded_files = set([doc.metadata.get('source') for doc in documents])

print(f"\nУспешно считано уникальных файлов: {len(loaded_files)}\n")
print("Список распознанных файлов:")
for file in loaded_files:
    print(f" - {file}")