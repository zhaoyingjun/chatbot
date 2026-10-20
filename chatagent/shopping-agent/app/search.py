"""Anonymous Parallel Search MCP over Streamable HTTP (JSON and SSE responses)."""

import asyncio
import json
import os
from urllib.parse import urlsplit

import httpx

ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "chatbot-shopping-agent/1.0 (+https://github.com/zhaoyingjun/chatbot)"


class SearchError(Exception):
    pass


def search_provider():
    # An explicit disabled or unknown selection must never fall back.
    provider = os.getenv("CHATBOT_SEARCH_PROVIDER", "parallel").strip().lower()
    if provider != "parallel":
        raise SearchError(
            "网页检索已禁用" if provider == "none" else "不支持配置的网页检索服务"
        )
    return provider


def public_url(value):
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )
    except (TypeError, ValueError):
        return False


class ParallelSearch:
    async def call(self, name, arguments, conversation_id, model_name):
        search_provider()
        if name not in ("web_search", "web_fetch"):
            raise SearchError("未知网页工具")
        if name == "web_search":
            objective = arguments.get("objective")
            queries = arguments.get("search_queries")
            if (
                not isinstance(objective, str)
                or not objective.strip()
                or not isinstance(queries, list)
                or not 1 <= len(queries) <= 3
                or any(
                    not isinstance(q, str) or not q.strip() or len(q) > 500
                    for q in queries
                )
                or len(objective) > 1000
            ):
                raise SearchError("搜索需要 objective 和 1 至 3 个 search_queries")
            params = {"objective": objective, "search_queries": queries}
        else:
            urls = arguments.get("urls")
            if (
                not isinstance(urls, list)
                or not 1 <= len(urls) <= 3
                or any(not isinstance(u, str) or not public_url(u) for u in urls)
            ):
                raise SearchError("网页读取需要 1 至 3 个 HTTP/HTTPS URL")
            params = {"urls": urls}
        params["session_id"] = str(conversation_id)
        params["model_name"] = model_name
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/event-stream",
        }
        try:
            # Bound the entire discovery + execution, and propagate cancellation.
            return await asyncio.wait_for(self._call(name, params, headers), timeout=25)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            raise SearchError("网页检索超时，请稍后重试") from None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise SearchError("网页检索额度暂时用尽，请稍后重试") from None
            raise SearchError("网页检索服务请求失败") from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise SearchError("网页检索返回无效响应或连接失败") from None

    async def _call(self, name, params, headers):
        async with httpx.AsyncClient(headers=headers, timeout=20) as client:
            initialized, response_headers = await self._rpc(
                client,
                1,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "chatbot-shopping-agent", "version": "1.0"},
                },
            )
            client.headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
            if response_headers.get("mcp-session-id"):
                client.headers["Mcp-Session-Id"] = response_headers["mcp-session-id"]
            await self._rpc(client, None, "notifications/initialized", {})
            discovered, _ = await self._rpc(client, 2, "tools/list", {})
            if name not in {tool["name"] for tool in discovered["tools"]}:
                raise SearchError("网页检索服务缺少所需工具")
            result, _ = await self._rpc(
                client, 3, "tools/call", {"name": name, "arguments": params}
            )
            if result.get("isError"):
                raise SearchError("网页工具执行失败，请稍后重试")
            texts = [
                c["text"] for c in result.get("content", []) if c.get("type") == "text"
            ]
            if not texts:
                raise SearchError("网页检索未返回内容")
            return "\n".join(texts)[:16000]

    @staticmethod
    async def _rpc(client, request_id, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            payload["id"] = request_id
        async with client.stream("POST", ENDPOINT, json=payload) as response:
            response.raise_for_status()
            if request_id is None:
                return {}, response.headers
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > 1024 * 1024:
                    raise SearchError("网页检索响应过大")
            content = raw.decode("utf-8")
            if "text/event-stream" in response.headers.get("content-type", ""):
                messages = []
                for event in content.replace("\r\n", "\n").split("\n\n"):
                    data = "\n".join(
                        line[5:].lstrip(" ")
                        for line in event.splitlines()
                        if line.startswith("data:")
                    )
                    if data:
                        messages.append(json.loads(data))
                message = next((m for m in messages if m.get("id") == request_id), None)
                if message is None:
                    raise ValueError("Missing MCP response")
            else:
                message = json.loads(content)
            if message.get("id") != request_id or message.get("error"):
                raise SearchError("网页检索协议错误")
            return message["result"], response.headers
