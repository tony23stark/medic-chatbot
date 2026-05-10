@echo off
REM Activate virtualenv and run Streamlit from project root
pushd "%~dp0"
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo Virtual environment activation script not found at .venv\Scripts\activate.bat
  echo Activate your venv manually or edit this script.
)
python -m streamlit run medibot.py --server.port 8502
popd