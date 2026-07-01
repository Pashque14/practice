from langchain_community.llms import Ollama

# Подключаемся к нашей созданной модели
llm = Ollama(model="saiga")

print("Ожидаем ответ от модели...\n")
# Отправляем тестовый запрос
response = llm.invoke("Объясни в двух предложениях, зачем нужно логирование в IT-системах?")

print(response)