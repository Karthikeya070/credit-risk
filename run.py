# run.py
import os
import subprocess

port = int(os.environ.get("PORT", 7860))

subprocess.run([
    "streamlit",
    "run",
    "app.py",
    "--server.port",
    str(port),
    "--server.address",
    "0.0.0.0"
])
