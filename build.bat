@echo off
setlocal

echo [1] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist py2cpp.egg-info rmdir /s /q py2cpp.egg-info

echo [2] Building source and wheel distributions...
python setup.py sdist bdist_wheel
IF %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    exit /b 1
)

echo [3] Uploading package via twine...
python -m twine upload dist/*

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist py2cpp.egg-info rmdir /s /q py2cpp.egg-info

IF %ERRORLEVEL% NEQ 0 (
    echo Upload failed.
    exit /b 1
)

echo Done.
endlocal