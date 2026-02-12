# run_app.py

import subprocess

print("========================================")
print("  Marathon Predictor - Aplikacja")
print("========================================\n")
print("Uruchamianie aplikacji Streamlit...\n")
print("Aplikacja otworzy się w przeglądarce pod adresem:")
print("http://localhost:8501\n")
print("Aby zakończyć aplikację, naciśnij CTRL+C")
print("========================================\n")

subprocess.run(["streamlit", "run", "app.py"])
