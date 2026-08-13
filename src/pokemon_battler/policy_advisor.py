from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pokemon_battler.live_policy import InteractionPolicyRuntime


class PolicyAdvisorServer:
    """Expose one loaded Qwen policy to an isolated local teacher process."""

    def __init__(
        self,
        runtime: InteractionPolicyRuntime,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.runtime = runtime
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    row = json.loads(self.rfile.read(size))
                    with owner.lock:
                        if row.get("decision_phase") == "team_preview":
                            action_id = owner.runtime.predict_team_preview(row)
                            result: dict[str, Any] = {"action_id": action_id}
                        else:
                            prediction = owner.runtime.predict(row)
                            result = {
                                "action_id": prediction.action_id,
                                "preferences": prediction.preferences,
                                "value_probability": prediction.value_probability,
                            }
                    payload = json.dumps(result, separators=(",", ":")).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as error:  # noqa: BLE001 - return inference errors to caller
                    payload = json.dumps(
                        {"error": f"{type(error).__name__}: {error}"}
                    ).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="qwen-policy-advisor", daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/predict"

    def __enter__(self) -> PolicyAdvisorServer:  # noqa: PYI034
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
