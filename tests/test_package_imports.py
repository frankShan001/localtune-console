import subprocess
import sys
from pathlib import Path


def test_constants_import_does_not_load_training_dependencies():
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "import src.constants; "
        "assert 'datasets' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=project_root, check=True)
