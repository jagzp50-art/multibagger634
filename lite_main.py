"""
Sovereign Lite v11 — application entrypoint.

Run:  python3 lite_main.py
Serves the 5-screen dashboard + JSON API on :9005 (API_PORT / PORT aware).
"""
import os

from lite.api import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT") or os.getenv("PORT") or 9005)
    host = os.getenv("API_HOST", "0.0.0.0")
    print(f"[lite] Sovereign Lite v11 — http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
