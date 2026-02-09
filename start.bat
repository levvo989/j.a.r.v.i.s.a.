@echo off
REM Создаём виртуальное окружение, если его нет
if not exist venv (
    python -m venv venv
)

REM Активируем виртуальное окружение
call venv\Scripts\activate

REM Устанавливаем зависимости
pip install --upgrade pip
pip install -r requirements.txt

REM Запускаем Flask сервер
python app.py

REM Оставляем окно открытым после завершения
echo.
echo Нажмите любую клавишу для выхода...
pause
