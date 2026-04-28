@echo off
echo Starting the Ellipse Detection pipeline...

:: Check if virtual environment exists, create if not
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Install required libraries (quietly to avoid spamming the console)
echo Installing dependencies...
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

:: Run the main program
echo Running the script...
python main.py

:: Keep the terminal open when finished so the user can read the output
pause