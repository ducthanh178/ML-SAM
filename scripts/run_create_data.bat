@echo off
echo Creating sample data files...
python scripts/create_data.py
if %errorlevel% neq 0 (
    echo.
    echo Python not found. Please install Python or add it to PATH.
    echo Alternatively, you can manually run: python scripts/create_data.py
    pause
) else (
    echo.
    echo ✅ Successfully created all JSON files!
    echo.
    echo Note: To create loss_surface.npy files, you need numpy installed.
    echo Run: python scripts/generate_sample_data.py
    pause
)





