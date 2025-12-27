import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def setup_logger(assets_folder: Path, run_id: str, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging.

    Creates a logs folder under assets and writes to pipeline_{run_id}.log.
    """
    logs_dir = assets_folder / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"etl_similarities:{run_id}")
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates on multiple runs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(logs_dir / f"pipeline_{run_id}.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class PipelineTracker:
    """Tracks step status and writes a JSON status file.

    The status file is always written to assets_folder/pipeline_status.json.
    Each step contains: status, start_time, end_time, duration_seconds, extra.
    """

    def __init__(self, assets_folder: Path, run_id: Optional[str] = None):
        self.assets_folder = assets_folder
        self.status_path = assets_folder / "pipeline_status.json"
        self.data: Dict[str, Dict] = {
            "run_id": run_id or datetime.now().strftime("%Y%m%d-%H%M%S"),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "steps": {},
        }

    def start(self, step: str, extra: Optional[Dict] = None):
        self.data["steps"].setdefault(step, {})
        self.data["steps"][step].update(
            {
                "status": "running",
                "start_time": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if extra:
            self.merge_extra(step, extra)
        self._write()

    def end(self, step: str, success: bool = True, extra: Optional[Dict] = None):
        info = self.data["steps"].setdefault(step, {})
        info["end_time"] = datetime.now().isoformat(timespec="seconds")
        # duration
        try:
            start_dt = datetime.fromisoformat(info.get("start_time"))
            end_dt = datetime.fromisoformat(info["end_time"])
            info["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 2)
        except Exception:
            info["duration_seconds"] = None
        info["status"] = "completed" if success else "failed"
        if extra:
            self.merge_extra(step, extra)
        self._write()

    def fail(self, step: str, error_message: str, extra: Optional[Dict] = None):
        self.end(step, success=False, extra={"error": error_message, **(extra or {})})

    def merge_extra(self, step: str, extra: Dict):
        info = self.data["steps"].setdefault(step, {})
        extra_prev = info.get("extra", {})
        extra_prev.update(extra)
        info["extra"] = extra_prev

    def _write(self):
        try:
            with open(self.status_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            # Avoid hard crashes due to status file issues
            pass

    def completed_steps(self) -> set:
        return {s for s, d in self.data.get("steps", {}).items() if d.get("status") == "completed"}
