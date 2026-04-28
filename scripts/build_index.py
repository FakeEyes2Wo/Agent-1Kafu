from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kefu_agent.config import get_settings
from kefu_agent.rag import build_index


if __name__ == "__main__":
    count = build_index()
    print(f"indexed_chunks={count}")
    print(f"index_path={get_settings().index_path}")
