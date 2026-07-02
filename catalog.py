import json

def load_catalog(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        return catalog
    except Exception:
        return []
