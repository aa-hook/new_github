"""Local forwarding proxy that counts every byte sent through an upstream proxy."""

from __future__ import annotations

import base64
import ipaddress
import socket
import socketserver
import ssl
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import unquote, urlsplit, urlunsplit


MAX_HEADER_BYTES = 128 * 1024
BUFFER_SIZE = 64 * 1024


@dataclass(frozen=True)
class UpstreamProxy:
    scheme: str
    host: str
    port: int
    username: Optional[str]
    password: Optional[str]

    @classmethod
    def from_url(cls, value: str) -> "UpstreamProxy":
        parsed = urlsplit(str(value or "").strip())
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https", "socks5", "socks5h"}:
            raise ValueError(f"unsupported metered proxy scheme: {scheme}")
        host = parsed.hostname or ""
        if not host:
            raise ValueError("metered proxy host is empty")
        if parsed.port is None:
            raise ValueError("metered proxy port is required")
        username = unquote(parsed.username) if parsed.username is not None else None
        password = unquote(parsed.password) if parsed.password is not None else None
        if (username is None) != (password is None):
            raise ValueError("metered proxy username and password must be paired")
        return cls(scheme, host, int(parsed.port), username, password)

    @property
    def display(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}"

    @property
    def basic_authorization(self) -> Optional[str]:
        if self.username is None:
            return None
        raw = f"{self.username}:{self.password or ''}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")


class TrafficCounters:
    def __init__(self, upstream: UpstreamProxy) -> None:
        self.upstream = upstream
        self.started_at = time.time()
        self.stopped_at: Optional[float] = None
        self.upload_bytes = 0
        self.download_bytes = 0
        self.connections = 0
        self.active_connections = 0
        self.failures = 0
        self.errors: list[str] = []
        self.targets: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)

    def _target_locked(self, target: str) -> dict[str, int]:
        key = str(target or "unclassified")
        return self.targets.setdefault(
            key,
            {
                "uploadBytes": 0,
                "downloadBytes": 0,
                "connections": 0,
                "activeConnections": 0,
                "failures": 0,
            },
        )

    def add_upload(self, size: int, target: str = "unclassified") -> None:
        if size > 0:
            with self._lock:
                self.upload_bytes += int(size)
                self._target_locked(target)["uploadBytes"] += int(size)

    def add_download(self, size: int, target: str = "unclassified") -> None:
        if size > 0:
            with self._lock:
                self.download_bytes += int(size)
                self._target_locked(target)["downloadBytes"] += int(size)

    def add_connection(self, target: str = "unclassified") -> None:
        with self._lock:
            self.connections += 1
            self.active_connections += 1
            target_counter = self._target_locked(target)
            target_counter["connections"] += 1
            target_counter["activeConnections"] += 1

    def finish_connection(self, target: str = "unclassified") -> None:
        with self._idle:
            self.active_connections = max(0, self.active_connections - 1)
            target_counter = self._target_locked(target)
            target_counter["activeConnections"] = max(
                0,
                target_counter["activeConnections"] - 1,
            )
            self._idle.notify_all()

    def wait_for_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._idle:
            while self.active_connections:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def add_failure(
        self,
        exc: BaseException,
        target: str = "unclassified",
    ) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if self.upstream.username:
            message = message.replace(self.upstream.username, "<redacted>")
        if self.upstream.password:
            message = message.replace(self.upstream.password, "<redacted>")
        with self._lock:
            self.failures += 1
            self._target_locked(target)["failures"] += 1
            self.errors.append(message[:500])
            del self.errors[:-10]

    def snapshot(self, *, local_url: str) -> dict[str, Any]:
        with self._lock:
            upload = int(self.upload_bytes)
            download = int(self.download_bytes)
            connections = int(self.connections)
            active_connections = int(self.active_connections)
            failures = int(self.failures)
            errors = list(self.errors)
            targets = {
                key: dict(value)
                for key, value in self.targets.items()
            }
            stopped = self.stopped_at
        total = upload + download
        ended = stopped or time.time()
        target_rows = []
        for target, value in targets.items():
            target_upload = int(value.get("uploadBytes") or 0)
            target_download = int(value.get("downloadBytes") or 0)
            target_total = target_upload + target_download
            target_rows.append(
                {
                    "target": target,
                    "uploadBytes": target_upload,
                    "downloadBytes": target_download,
                    "totalBytes": target_total,
                    "uploadMiB": round(target_upload / (1024 * 1024), 4),
                    "downloadMiB": round(target_download / (1024 * 1024), 4),
                    "totalMiB": round(target_total / (1024 * 1024), 4),
                    "connections": int(value.get("connections") or 0),
                    "activeConnections": int(value.get("activeConnections") or 0),
                    "failures": int(value.get("failures") or 0),
                }
            )
        target_rows.sort(key=lambda item: int(item["totalBytes"]), reverse=True)
        return {
            "enabled": True,
            "upstreamProxy": self.upstream.display,
            "localMeterProxy": local_url,
            "uploadBytes": upload,
            "downloadBytes": download,
            "totalBytes": total,
            "uploadMiB": round(upload / (1024 * 1024), 4),
            "downloadMiB": round(download / (1024 * 1024), 4),
            "totalMiB": round(total / (1024 * 1024), 4),
            "connections": connections,
            "activeConnections": active_connections,
            "failures": failures,
            "errors": errors,
            "targets": target_rows,
            "durationSeconds": round(max(0.0, ended - self.started_at), 3),
            "measurement": (
                "bytes read from and written to the upstream proxy socket; "
                "includes CONNECT/SOCKS handshakes and tunneled TLS, excludes TCP/IP framing"
            ),
        }


def _read_until_header(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(BUFFER_SIZE)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_HEADER_BYTES:
            raise ValueError("proxy request header exceeds limit")
    return bytes(data)


def _split_header(data: bytes) -> tuple[bytes, bytes]:
    marker = data.find(b"\r\n\r\n")
    if marker < 0:
        raise ValueError("incomplete proxy request header")
    marker += 4
    return data[:marker], data[marker:]


def _request_line(header: bytes) -> tuple[str, str, str]:
    line = header.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
    parts = line.split(" ", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid proxy request line: {line!r}")
    return parts[0].upper(), parts[1], parts[2]


def _inject_proxy_authorization(data: bytes, authorization: Optional[str]) -> bytes:
    header, remainder = _split_header(data)
    lines = header[:-4].split(b"\r\n")
    filtered = [
        line
        for line in lines
        if not line.lower().startswith(b"proxy-authorization:")
    ]
    if authorization:
        filtered.append(f"Proxy-Authorization: {authorization}".encode("latin-1"))
    return b"\r\n".join(filtered) + b"\r\n\r\n" + remainder


def _origin_form_request(data: bytes) -> tuple[bytes, str, int]:
    header, remainder = _split_header(data)
    lines = header[:-4].split(b"\r\n")
    method, target, version = _request_line(header)
    parsed = urlsplit(target) if "://" in target else None
    host = parsed.hostname if parsed else None
    port = parsed.port if parsed and parsed.port else (443 if parsed and parsed.scheme == "https" else 80)
    if parsed:
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    else:
        path = target
        for line in lines[1:]:
            if line.lower().startswith(b"host:"):
                host_value = line.split(b":", 1)[1].strip().decode("latin-1")
                host, port = _split_host_port(host_value, 80)
                break
    if not host:
        raise ValueError("plain HTTP proxy request has no target host")
    cleaned = [lines[0].split(b" ", 1)[0] + b" " + path.encode("latin-1") + b" " + version.encode("ascii")]
    cleaned.extend(
        line
        for line in lines[1:]
        if not line.lower().startswith((b"proxy-authorization:", b"proxy-connection:"))
    )
    return b"\r\n".join(cleaned) + b"\r\n\r\n" + remainder, host, int(port)


def _split_host_port(value: str, default_port: int) -> tuple[str, int]:
    text = str(value or "").strip()
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            raise ValueError(f"invalid IPv6 target: {text!r}")
        host = text[1:end]
        tail = text[end + 1 :]
        return host, int(tail[1:]) if tail.startswith(":") else default_port
    if text.count(":") == 1:
        host, port = text.rsplit(":", 1)
        return host, int(port)
    return text, default_port


def _format_target(host: str, port: int) -> str:
    normalized_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"{normalized_host}:{int(port)}"


class ProxyTrafficMeter:
    def __init__(
        self,
        upstream_url: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        connect_timeout: float = 30.0,
    ) -> None:
        self.upstream = UpstreamProxy.from_url(upstream_url)
        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self.counters = TrafficCounters(self.upstream)
        self.server: Optional[_MeterServer] = None
        self.thread: Optional[threading.Thread] = None
        self.local_url = ""

    def start(self) -> str:
        if self.server is not None:
            return self.local_url
        server = _MeterServer((self.host, self.port), _MeterHandler)
        server.meter = self
        self.server = server
        bound_host, bound_port = server.server_address[:2]
        self.local_url = f"http://{bound_host}:{bound_port}"
        self.thread = threading.Thread(
            target=server.serve_forever,
            name="proxy-traffic-meter",
            daemon=True,
        )
        self.thread.start()
        return self.local_url

    def stop(self) -> dict[str, Any]:
        if self.server is not None:
            self.server.shutdown()
            self.counters.wait_for_idle(3.0)
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        self.counters.stopped_at = time.time()
        self.server = None
        self.thread = None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return self.counters.snapshot(local_url=self.local_url)

    def _connect_http_upstream(self) -> socket.socket:
        raw = socket.create_connection(
            (self.upstream.host, self.upstream.port),
            timeout=self.connect_timeout,
        )
        if self.upstream.scheme == "https":
            context = ssl.create_default_context()
            raw = context.wrap_socket(raw, server_hostname=self.upstream.host)
        raw.settimeout(None)
        return raw

    def _send_upstream(
        self,
        sock: socket.socket,
        data: bytes,
        target: str = "unclassified",
    ) -> None:
        if not data:
            return
        sock.sendall(data)
        self.counters.add_upload(len(data), target)

    def _recv_upstream(
        self,
        sock: socket.socket,
        size: int = BUFFER_SIZE,
        target: str = "unclassified",
    ) -> bytes:
        data = sock.recv(size)
        self.counters.add_download(len(data), target)
        return data

    def _recv_exact_upstream(
        self,
        sock: socket.socket,
        size: int,
        target: str = "unclassified",
    ) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._recv_upstream(sock, size - len(chunks), target)
            if not chunk:
                raise ConnectionError("upstream proxy closed during handshake")
            chunks.extend(chunk)
        return bytes(chunks)

    def _connect_socks_target(
        self,
        target_host: str,
        target_port: int,
        target: str,
    ) -> socket.socket:
        sock = socket.create_connection(
            (self.upstream.host, self.upstream.port),
            timeout=self.connect_timeout,
        )
        methods = b"\x02\x00" if self.upstream.username is not None else b"\x00"
        self._send_upstream(sock, b"\x05" + bytes([len(methods)]) + methods, target)
        selected = self._recv_exact_upstream(sock, 2, target)
        if selected[0] != 5 or selected[1] == 0xFF:
            raise ConnectionError(f"SOCKS5 authentication method rejected: {selected!r}")
        if selected[1] == 2:
            username = (self.upstream.username or "").encode("utf-8")
            password = (self.upstream.password or "").encode("utf-8")
            if len(username) > 255 or len(password) > 255:
                raise ValueError("SOCKS5 credentials exceed 255 bytes")
            self._send_upstream(
                sock,
                b"\x01" + bytes([len(username)]) + username + bytes([len(password)]) + password,
                target,
            )
            auth_reply = self._recv_exact_upstream(sock, 2, target)
            if auth_reply[1] != 0:
                raise ConnectionError("SOCKS5 username/password authentication failed")
        elif selected[1] != 0:
            raise ConnectionError(f"unsupported SOCKS5 method: {selected[1]}")

        try:
            address = ipaddress.ip_address(target_host)
        except ValueError:
            encoded = target_host.encode("idna")
            if len(encoded) > 255:
                raise ValueError("SOCKS5 target hostname exceeds 255 bytes")
            address_field = b"\x03" + bytes([len(encoded)]) + encoded
        else:
            address_field = (b"\x01" if address.version == 4 else b"\x04") + address.packed
        self._send_upstream(
            sock,
            b"\x05\x01\x00" + address_field + int(target_port).to_bytes(2, "big"),
            target,
        )
        reply = self._recv_exact_upstream(sock, 4, target)
        if reply[0] != 5 or reply[1] != 0:
            raise ConnectionError(f"SOCKS5 CONNECT failed with code {reply[1]}")
        address_size = {1: 4, 4: 16}.get(reply[3])
        if reply[3] == 3:
            address_size = self._recv_exact_upstream(sock, 1, target)[0]
        if address_size is None:
            raise ConnectionError(f"invalid SOCKS5 reply address type: {reply[3]}")
        self._recv_exact_upstream(sock, address_size + 2, target)
        sock.settimeout(None)
        return sock

    def _read_http_upstream_header(
        self,
        sock: socket.socket,
        target: str,
    ) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self._recv_upstream(sock, target=target)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_HEADER_BYTES:
                raise ValueError("upstream proxy response header exceeds limit")
        return bytes(data)

    def _tunnel(
        self,
        client: socket.socket,
        upstream: socket.socket,
        target: str,
    ) -> None:
        def upload() -> None:
            try:
                while True:
                    data = client.recv(BUFFER_SIZE)
                    if not data:
                        break
                    self._send_upstream(upstream, data, target)
            except OSError:
                pass
            finally:
                try:
                    upstream.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        thread = threading.Thread(target=upload, daemon=True)
        thread.start()
        try:
            while True:
                data = self._recv_upstream(upstream, target=target)
                if not data:
                    break
                client.sendall(data)
        except OSError:
            pass
        finally:
            try:
                client.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            thread.join(timeout=2.0)

    def handle_client(self, client: socket.socket) -> None:
        upstream: Optional[socket.socket] = None
        connection_target = "unclassified"
        connection_started = False
        try:
            initial = _read_until_header(client)
            header, _ = _split_header(initial)
            method, target, _version = _request_line(header)
            if method == "CONNECT":
                target_host, target_port = _split_host_port(target, 443)
            else:
                _cleaned, target_host, target_port = _origin_form_request(initial)
            connection_target = _format_target(target_host, target_port)
            self.counters.add_connection(connection_target)
            connection_started = True
            if self.upstream.scheme in {"http", "https"}:
                upstream = self._connect_http_upstream()
                forwarded = _inject_proxy_authorization(
                    initial, self.upstream.basic_authorization
                )
                self._send_upstream(upstream, forwarded, connection_target)
                if method == "CONNECT":
                    response = self._read_http_upstream_header(
                        upstream,
                        connection_target,
                    )
                    client.sendall(response)
                    status_line = response.split(b"\r\n", 1)[0]
                    if b" 2" not in status_line:
                        return
                self._tunnel(client, upstream, connection_target)
                return

            if method == "CONNECT":
                upstream = self._connect_socks_target(
                    target_host,
                    target_port,
                    connection_target,
                )
                client.sendall(
                    b"HTTP/1.1 200 Connection Established\r\n"
                    b"Proxy-Agent: V4TrafficMeter\r\n\r\n"
                )
            else:
                initial, target_host, target_port = _origin_form_request(initial)
                upstream = self._connect_socks_target(
                    target_host,
                    target_port,
                    connection_target,
                )
                self._send_upstream(upstream, initial, connection_target)
            self._tunnel(client, upstream, connection_target)
        except Exception as exc:
            self.counters.add_failure(exc, connection_target)
            try:
                client.sendall(
                    b"HTTP/1.1 502 Bad Gateway\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            except OSError:
                pass
        finally:
            if upstream is not None:
                try:
                    upstream.close()
                except OSError:
                    pass
            if connection_started:
                self.counters.finish_connection(connection_target)


class _MeterServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    meter: ProxyTrafficMeter


class _MeterHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.server.meter.handle_client(self.request)


__all__ = ["ProxyTrafficMeter", "TrafficCounters", "UpstreamProxy"]
