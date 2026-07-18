import base64
import socket
import socketserver
import threading
import time

from proxy_traffic_meter import ProxyTrafficMeter, UpstreamProxy


def recv_header(sock):
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


class EchoProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        header = recv_header(self.request)
        self.server.headers.append(header)
        self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        while True:
            data = self.request.recv(65536)
            if not data:
                break
            self.server.payloads.append(data)
            self.request.sendall(b"ECHO:" + data)


class EchoProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, EchoProxyHandler)
        self.headers = []
        self.payloads = []


def test_upstream_proxy_parsing_redacts_credentials():
    upstream = UpstreamProxy.from_url(
        "http://user%20name:p%40ss@proxy.example:3128"
    )

    assert upstream.username == "user name"
    assert upstream.password == "p@ss"
    assert upstream.display == "http://proxy.example:3128"
    assert "user" not in upstream.display
    expected = base64.b64encode(b"user name:p@ss").decode("ascii")
    assert upstream.basic_authorization == f"Basic {expected}"


def test_http_connect_is_forwarded_and_all_upstream_socket_bytes_are_counted():
    upstream = EchoProxyServer(("127.0.0.1", 0))
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address
    meter = ProxyTrafficMeter(f"http://user:secret@{host}:{port}")
    local_url = meter.start()
    local = local_url.removeprefix("http://").split(":")

    client = socket.create_connection((local[0], int(local[1])), timeout=5)
    client.sendall(
        b"CONNECT target.example:443 HTTP/1.1\r\n"
        b"Host: target.example:443\r\n\r\n"
    )
    response = recv_header(client)
    assert b" 200 " in response
    client.sendall(b"HELLO-METER")
    assert client.recv(65536) == b"ECHO:HELLO-METER"
    client.shutdown(socket.SHUT_RDWR)
    client.close()

    deadline = time.time() + 2
    while time.time() < deadline and not upstream.payloads:
        time.sleep(0.01)
    report = meter.stop()
    upstream.shutdown()
    upstream.server_close()
    upstream_thread.join(timeout=2)

    assert len(upstream.headers) == 1
    assert b"Proxy-Authorization: Basic dXNlcjpzZWNyZXQ=" in upstream.headers[0]
    assert upstream.payloads == [b"HELLO-METER"]
    assert report["uploadBytes"] >= len(upstream.headers[0]) + len(b"HELLO-METER")
    assert report["downloadBytes"] >= len(response) + len(b"ECHO:HELLO-METER")
    assert report["totalBytes"] == report["uploadBytes"] + report["downloadBytes"]
    assert report["connections"] == 1
    assert report["activeConnections"] == 0
    assert report["failures"] == 0
    assert report["upstreamProxy"] == f"http://{host}:{port}"
    assert len(report["targets"]) == 1
    target = report["targets"][0]
    assert target["target"] == "target.example:443"
    assert target["uploadBytes"] == report["uploadBytes"]
    assert target["downloadBytes"] == report["downloadBytes"]
    assert target["totalBytes"] == report["totalBytes"]
    assert target["connections"] == 1
    assert target["activeConnections"] == 0
    assert target["failures"] == 0
    assert "user" not in str(report)
    assert "secret" not in str(report)
