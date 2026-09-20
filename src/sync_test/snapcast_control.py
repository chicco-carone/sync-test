"""Small synchronous client for Snapserver's documented JSON-RPC control API."""

import json
import socket


class SnapcastRPCError(RuntimeError):
    """A Snapserver JSON-RPC request failed."""


class SnapcastController:
    """Control one Snapclient through Snapserver's newline-delimited RPC port."""

    def __init__(self, host: str, port: int = 1705, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.request_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.request_id += 1
        request = {"id": self.request_id, "jsonrpc": "2.0", "method": method}
        if params is not None:
            request["params"] = params

        with socket.create_connection((self.host, self.port), self.timeout) as connection:
            connection.settimeout(self.timeout)
            connection.sendall((json.dumps(request) + "\n").encode())
            response = b""
            while not response.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    raise SnapcastRPCError("Snapserver closed the RPC connection without a response")
                response += chunk

        payload = json.loads(response)
        if "error" in payload:
            error = payload["error"]
            raise SnapcastRPCError(f"{method}: {error.get('message', error)}")
        return payload["result"]

    def get_client(self, client_id: str) -> dict:
        return self.call("Client.GetStatus", {"id": client_id})["client"]

    def set_latency(self, client_id: str, latency_ms: int) -> int:
        result = self.call("Client.SetLatency", {"id": client_id, "latency": latency_ms})
        return result["latency"]

    def force_group_rejoin(self, client_id: str) -> None:
        """Remove then restore a client to its current group.

        Snapserver has no dedicated client-resync RPC method. Reapplying group
        membership is the least invasive API-only way to make it receive the
        group and stream configuration again.
        """
        server = self.call("Server.GetStatus")["server"]
        for group in server["groups"]:
            client_ids = [client["id"] for client in group["clients"]]
            if client_id in client_ids:
                if len(client_ids) == 1:
                    raise SnapcastRPCError(
                        "Cannot safely force a group rejoin when the target is the group's only client"
                    )
                self.call("Group.SetClients", {
                    "id": group["id"],
                    "clients": [item for item in client_ids if item != client_id],
                })
                self.call("Group.SetClients", {"id": group["id"], "clients": client_ids})
                return
        raise SnapcastRPCError(f"Client {client_id!r} is not assigned to a group")
