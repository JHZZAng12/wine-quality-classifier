"""프로젝트 안에서 사용하는 경로를 한곳에 모았습니다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'
FIGURES_DIR = ROOT / 'figures'
MODELS_DIR = ROOT / 'models'
EXAMPLES_DIR = ROOT / 'examples'
DOCS_DIR = ROOT / 'docs'

def ensure_output_dirs():
    for folder in [DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, EXAMPLES_DIR, DOCS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
