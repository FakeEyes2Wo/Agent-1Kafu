from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from kefu_agent.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "kefu_agent.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
