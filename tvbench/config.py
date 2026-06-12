from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(profile: str) -> dict:
    path = ROOT / "config" / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as handle:
        config = yaml.safe_load(handle)

    for key in ("data_dir", "index_dir", "run_dir"):
        value = Path(config["paths"][key])
        if not value.is_absolute():
            value = ROOT / value
        config["paths"][key] = value

    return config


def ensure_run_dirs(config: dict) -> None:
    run_dir: Path = config["paths"]["run_dir"]
    for name in ("state", "logs", "results", "charts", "artifacts"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    config["paths"]["data_dir"].mkdir(parents=True, exist_ok=True)
    config["paths"]["index_dir"].mkdir(parents=True, exist_ok=True)
