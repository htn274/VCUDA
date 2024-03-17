from pathlib import Path

PROJECT_NAME = 'BCD'
PROJECT_ROOT = Path(__file__).absolute().parent.parent
EXP_DIR = PROJECT_ROOT / 'exp'
DATA_DIR = PROJECT_ROOT / 'data'
SAVE_DIR = PROJECT_ROOT / 'save'
REAL_DATA = ['sachs', 'syntren']