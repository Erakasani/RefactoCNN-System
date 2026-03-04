from __future__ import annotations
import argparse
from src.refactocnn.ui.flask_app import create_app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
