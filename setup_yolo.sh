#!/bin/bash

echo "🟩 Починаємо встановлення YOLOv5-проєкту..."

# Назва віртуального оточення
ENV_NAME="yolov5-env"

# Перевірка наявності Python 3.10+ (або інший бажаний)
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 не знайдено. Встанови Python 3 перед продовженням."
    exit 1
fi

# Створення віртуального середовища
echo "🔧 Створюємо віртуальне середовище '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"

# Активація віртуального середовища
echo "⚙️ Активуємо віртуальне середовище..."
source "$ENV_NAME/bin/activate"

# Перевірка наявності файлу requirements.txt у цій же директорії
REQ_FILE="$(dirname "$0")/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "❌ Помилка: файл 'requirements.txt' не знайдено в поточній папці."
    echo "   Поточна директорія: $(pwd)"
    echo "   Очікуваний шлях: $REQ_FILE"
    echo "➡️ Спочатку створи requirements.txt перед запуском цього скрипта."
    deactivate
    exit 1
fi

# Встановлення залежностей
echo "📦 Встановлюємо залежності з $REQ_FILE ..."
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo "✅ Установка завершена."
echo "👉 Перед запуском активація середовища: source $ENV_NAME/bin/activate"
