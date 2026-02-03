# environment/executor.py
import subprocess
import tempfile
import sys

class Executor:
    @staticmethod
    def run_code(script: str):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            filename = f.name

        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return False, result.stderr

        return True, result.stdout
