#!/bin/bash

# Оновлення системи
sudo apt update && sudo apt upgrade -y

# Встановлення необхідних системних пакетів
sudo apt install -y python3 python3-pip python3-venv git wget build-essential

# Створення та активація віртуального оточення
python3 -m venv yolov5-env
source yolov5-env/bin/activate

# Оновлення pip
pip install --upgrade pip

# Встановлення PyTorch з підтримкою CUDA 12.1 (під RTX 4050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Клонування репозиторію YOLOv5 (якщо його ще немає)
if [ ! -d "yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5.git
fi

cd yolov5

# Встановлення залежностей YOLOv5
pip install -r requirements.txt

echo "Установка завершена. Активуй віртуальне оточення командою 'source yolov5-env/bin/activate' перед запуском."
