from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import aiohttp

from aiotcvectordb import exceptions
from aiotcvectordb.exceptions import ParamError, ServerInternalError


class Response:
    def __init__(
        self,
        path: str,
        json_body: Dict[str, Any],
        status: int,
        reason: str,
        warn_header: Optional[str] = None,
    ):
        """HTTP Response wrapper to align with sync client's interface."""
        # When HTTP status indicates error and body isn't the standard code/msg, raise
        if status >= 400 and ("code" not in json_body or "msg" not in json_body):
            message = json_body if isinstance(json_body, str) else str(json_body)
            raise ServerInternalError(code=status, message=f"{reason}: {message}")

        self._body = json_body
        self._code = int(json_body.get("code", 0))
        self._message = json_body.get("msg", "")
        self.req_id = json_body.get("requestId", None)
        self._warn = warn_header

    @property
    def code(self) -> int:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    @property
    def body(self) -> Dict[str, Any]:
        return self._body

    def data(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        res.update(self._body)
        return res


class AsyncHTTPClient:
    def __init__(
        self,
        url: str,
        username: str,
        key: str,
        timeout: int = 10,
        pool_size: int = 10,
        proxies: Optional[dict] = None,
        password: Optional[str] = None,
        connector: Optional[aiohttp.BaseConnector] = None,
    ):
        self.url = url
        self.username = username
        self.key = key
        self.password = password
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self._authorization()}",
        }
        self._proxies = proxies or {}
        self.direct = False
        self._pool_size = pool_size
        self._connector = connector
        # 会话延迟创建，确保在事件循环中实例化，避免非 ioloop 报错
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> None:
        if self._session and not self._session.closed:
            return
        timeout_obj = aiohttp.ClientTimeout(
            total=None if (self.timeout is None or self.timeout <= 0) else self.timeout
        )
        connector = self._connector or aiohttp.TCPConnector(
            limit=self._pool_size,
            ttl_dns_cache=True,
        )
        self._session = aiohttp.ClientSession(timeout=timeout_obj, connector=connector)

    async def __aenter__(self) -> "AsyncHTTPClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def _authorization(self) -> str:
        if self.password is None:
            self.password = self.key
        if not self.username or not self.password:
            raise ParamError(
                message="Network or authentication settings are invalid, please check url/username/api_key."
            )
        return f"account={self.username}&api_key={self.password}"

    def _get_url(self, path: str) -> str:
        if not self.url:
            raise ParamError(
                message="Network or authentication settings are invalid, please check url/username/api_key."
            )
        return self.url + path

    def _get_headers(self, ai: Optional[bool] = False) -> Dict[str, str]:
        if ai is None:
            return dict(self._headers)
        backend = "vdb"
        if (not self.direct) and ai:
            backend = "ai"
        headers = {"backend-service": backend}
        headers.update(self._headers)
        return headers

    def _choose_proxy(self) -> Optional[str]:
        if not self._proxies:
            return None
        scheme = urlparse(self.url).scheme or "http"
        proxy = self._proxies.get(scheme)
        if proxy:
            return proxy
        # 回退：若未命中精确 scheme，则择其一常见代理
        return self._proxies.get("http") or self._proxies.get("https")

    async def get(
        self,
        path: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
        ai: Optional[bool] = False,
    ) -> Response:
        await self._ensure_session()
        # Per-request timeout overrides session's default
        timeout_ctx = aiohttp.ClientTimeout(
            total=None if (timeout is None or timeout <= 0) else timeout
        )
        proxy = self._choose_proxy()
        try:
            async with self._session.get(
                self._get_url(path),
                params=params,
                headers=self._get_headers(ai),
                proxy=proxy,
                timeout=timeout_ctx,
            ) as resp:
                warn = resp.headers.get("Warning")
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    text = await resp.text()
                    # attempt to construct a body for consistent error handling
                    body = {"code": resp.status, "msg": text}
                response = Response(path, body, resp.status, resp.reason, warn)
        except aiohttp.ClientConnectorError as e:
            raise exceptions.ConnectError(
                message=f"{e}: {exceptions.ERROR_MESSAGE_NETWORK_OR_AUTH}"
            )
        except aiohttp.ClientResponseError as e:
            raise ServerInternalError(code=e.status or -1, message=str(e))
        except asyncio.TimeoutError:
            raise ServerInternalError(code=-1, message="Request timed out")

        if response.code != 0:
            raise ServerInternalError(
                code=response.code, message=response.message, req_id=response.req_id
            )
        return response

    async def post(
        self,
        path: str,
        body: dict,
        timeout: Optional[float] = None,
        ai: Optional[bool] = False,
    ) -> Response:
        await self._ensure_session()
        timeout_ctx = aiohttp.ClientTimeout(
            total=None if (timeout is None or timeout <= 0) else timeout
        )
        proxy = self._choose_proxy()
        try:
            async with self._session.post(
                self._get_url(path),
                json=body,
                headers=self._get_headers(ai),
                proxy=proxy,
                timeout=timeout_ctx,
            ) as resp:
                warn = resp.headers.get("Warning")
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    text = await resp.text()
                    body = {"code": resp.status, "msg": text}
                response = Response(path, body, resp.status, resp.reason, warn)
        except aiohttp.ClientConnectorError as e:
            raise exceptions.ConnectError(
                message=f"{e}: {exceptions.ERROR_MESSAGE_NETWORK_OR_AUTH}"
            )
        except aiohttp.ClientResponseError as e:
            raise ServerInternalError(code=e.status or -1, message=str(e))
        except asyncio.TimeoutError:
            raise ServerInternalError(code=-1, message="Request timed out")

        if response.code != 0:
            raise ServerInternalError(
                code=response.code, message=response.message, req_id=response.req_id
            )
        return response

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
