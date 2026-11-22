"""Serve a generated viewer folder over HTTP (the viewers load data files, so file:// is not enough)."""

import argparse
import functools
import http.server

from ..paths import EVAL, RL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="add_sweetener")
    parser.add_argument("--dashboard", action="store_true", help="serve the RL dashboard instead")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    folder = RL / "dashboard" if args.dashboard else EVAL / args.task / "viewer"
    port = args.port or (8766 if args.dashboard else 8770)
    if not (folder / "index.html").exists():
        raise SystemExit(f"{folder} has no index.html: run {'pr-dashboard' if args.dashboard else 'pr-report'} first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))
    print(f"serving {folder} at http://localhost:{port}/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
