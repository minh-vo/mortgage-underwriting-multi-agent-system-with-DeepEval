"""Settings, constants and case loading. No other module reads the environment."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS = ROOT / "eval" / "results.json"  # written by eval/run_eval.py, read by the app
DATA_DIR = ROOT / "data"
POLICY_PATH = DATA_DIR / "policies.md"
CASES_PATH = DATA_DIR / "cases.json"

AREAS = ["credit", "income", "assets", "collateral"]
MAX_REWORKS = 3  # rework rounds the Supervisor may dispatch
MAX_FAILED_ATTEMPTS = 2  # after this many failures in a row an area is not retried

# US dollars per million tokens: (input, output). Unknown models show tokens only.
PRICES = {"claude-opus-5-5": (4.00, 20.00)}


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    judge_model: str
    critic_url: str
    db_path: str


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model=os.getenv("MODEL", "claude-opus-5-5"),
        judge_model=os.getenv("JUDGE_MODEL", "claude-opus-5-5"),
        critic_url=os.getenv("CRITIC_URL", "http://127.0.0.1:9105"),
        db_path=os.getenv("DB_PATH", str(ROOT / "underwriter.db")),
    )


def load_cases() -> dict[str, dict]:
    """Return {case_id: {"profile", "application", "expected"}}."""
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return {case["case_id"]: case for case in data["cases"]}


def load_application(case_id: str) -> dict:
    """Return only the application. The expected outcome never reaches an agent."""
    return load_cases()[case_id]["application"]


def usage_cost(usage: list[dict]) -> float | None:
    """Total cost in US dollars, or None when a model has no known price."""
    total = 0.0
    for record in usage:
        price = PRICES.get(record.get("model", ""))
        if price is None:
            return None
        total += record.get("input_tokens", 0) / 1e6 * price[0]
        total += record.get("output_tokens", 0) / 1e6 * price[1]
    return total
