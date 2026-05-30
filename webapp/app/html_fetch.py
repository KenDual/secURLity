"""
Async HTML fetcher with SSRF guard for SGD HTML-content model.
Returns (html_content | None, fetch_ms, error_message | None).
"""
import asyncio
import ipaddress
import socket
import time
from typing import Optional
from urllib.parse import urlsplit

import httpx

from .config import SGD_HTML_FETCH_TIMEOUT, SGD_MAX_HTML_BYTES

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
]
_PRIVATE_NETS_V6 = [
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_LOCALHOST_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


def _is_private_host(hostname: str) -> bool:
    """Block private IPs and localhost. Does direct IP check; hostname DNS is done async."""
    if hostname.lower() in _LOCALHOST_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.version == 4:
            return any(ip in net for net in _PRIVATE_NETS)
        else:
            return any(ip in net for net in _PRIVATE_NETS_V6)
    except ValueError:
        return False


async def _resolve_and_check(hostname: str) -> bool:
    """Returns True if hostname resolves to a private/loopback IP."""
    loop = asyncio.get_running_loop()
    try:
        ip_str = await loop.run_in_executor(None, socket.gethostbyname, hostname)
        return _is_private_host(ip_str)
    except Exception:
        return False


def _is_cloudflare_block(resp: httpx.Response) -> bool:
    if "Just a moment" in resp.text and "cloudflare" in resp.text.lower():
        return True
    if resp.status_code == 403 and "cf-mitigated" in resp.headers:
        return True
    return False


async def fetch_html(url: str) -> tuple[Optional[str], float, Optional[str]]:
    """
    Fetch HTML from url.
    Returns: (html_str | None, fetch_ms, error_label | None)
    error_label: "ssrf_blocked" | "cloudflare_blocked" | "timeout" | "fetch_failed"
    """
    t0 = time.perf_counter()

    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ""
    except Exception:
        return None, 0.0, "fetch_failed"

    if not hostname:
        return None, 0.0, "fetch_failed"

    # SSRF check: direct IP
    if _is_private_host(hostname):
        return None, 0.0, "ssrf_blocked"

    # SSRF check: DNS resolution
    if await _resolve_and_check(hostname):
        return None, 0.0, "ssrf_blocked"

    try:
        async with httpx.AsyncClient(
            timeout=SGD_HTML_FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=3,
            headers={"User-Agent": "Mozilla/5.0 secURLity-scanner/1.0"},
        ) as client:
            async with client.stream("GET", url) as resp:
                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= SGD_MAX_HTML_BYTES:
                        break
                raw = b"".join(chunks)

        fetch_ms = (time.perf_counter() - t0) * 1000

        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            html = raw.decode("latin-1", errors="replace")

        if _is_cloudflare_block(httpx.Response(resp.status_code, content=raw,
                                                headers=resp.headers)):
            return None, fetch_ms, "cloudflare_blocked"

        return html, fetch_ms, None

    except httpx.TimeoutException:
        return None, (time.perf_counter() - t0) * 1000, "timeout"
    except Exception:
        return None, (time.perf_counter() - t0) * 1000, "fetch_failed"
