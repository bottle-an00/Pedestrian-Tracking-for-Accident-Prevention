import yaml
import re
from pathlib import Path

VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[load_yaml] File not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    def substitute(value, ctx):
        if isinstance(value, str):
            matches = VAR_PATTERN.findall(value)
            for var in matches:
                if var in ctx:
                    value = value.replace("${" + var + "}", str(ctx[var]))
            return value

        elif isinstance(value, dict):
            return {k: substitute(v, ctx) for k, v in value.items()}

        elif isinstance(value, list):
            return [substitute(v, ctx) for v in value]

        return value  

    #(dataset_root 내부에서도 dataset_id가 쓰일 수 있음)
    data = substitute(data, data)
    data = substitute(data, data)

    return data
