@echo off
python -m pip install --upgrade pip

:: Instalacja bibliotek
pip install opencv-python numpy insightface onnxruntime-gpu keyboard colorama

:: Dodatkowe biblioteki dla działania insightface
pip install scipy tqdm psutil requests

echo.
pause
