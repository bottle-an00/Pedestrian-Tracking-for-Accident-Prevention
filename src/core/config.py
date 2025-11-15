# yaml 로더, 전역 설정 객체
import yaml
from pathlib import Path

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[load_yaml] File not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)