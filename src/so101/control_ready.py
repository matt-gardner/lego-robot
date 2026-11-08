"""Systemd startup acknowledgement after the broker's verified TLS listener is ready."""
import http.client
import json
import socket
import ssl
import time


def main() -> None:
    context = ssl.create_default_context(cafile="/etc/lego-robot-control/server.crt")
    deadline = time.monotonic() + 20
    while True:
        connection = http.client.HTTPSConnection("host.docker.internal", 8766, timeout=2)
        try:
            connection.sock = context.wrap_socket(
                socket.create_connection(("172.17.0.1", 8766), timeout=2,
                                         source_address=("127.0.0.1", 0)),
                server_hostname="host.docker.internal",
            )
            connection.request("POST", "/gpu.catalog", body="{}",
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = json.loads(response.read())
            if response.status != 403 or body != {"error": "Bearer capability required"}:
                raise RuntimeError("Unexpected broker readiness response")
            return
        except (ConnectionRefusedError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU broker TLS listener did not become ready")
            time.sleep(0.25)
        finally:
            connection.close()


if __name__ == "__main__":
    main()
