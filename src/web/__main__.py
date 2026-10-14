"""
    python -m src.web                     # http://127.0.0.1:8000, API docs at /docs
    python -m src.web --port 8080

Listens on this computer only (127.0.0.1): the development login is not
meant to be reachable from other machines.
"""

import argparse
import os

import uvicorn

from src.web.app import DEFAULT_DEV_PASSCODE, DEV_PASSCODE_ENV, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KORA's web API.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not os.environ.get(DEV_PASSCODE_ENV):
        print(f"DEVELOPMENT LOGIN: passcode '{DEFAULT_DEV_PASSCODE}' (set {DEV_PASSCODE_ENV} to change). "
              "Not production identity.")
    print("Loading KORA...")
    app = create_app()
    print(f"KORA API: http://127.0.0.1:{args.port}   (interactive docs: http://127.0.0.1:{args.port}/docs)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
