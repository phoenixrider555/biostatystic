

#!/bin/bash

chmod +x python1.py
echo "Установка зависимостей..."
pip install numpy pandas scipy

echo ""
echo "Запуск анализа..."
python3 python1.py
