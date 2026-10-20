"""Exercise the real /chat caller with scripted model and HTTP MCP responses."""

import asyncio
import ast
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID

import httpx

from app.chat_context import chat_context, reset_chat_context
from app.search import ENDPOINT, USER_AGENT
from app.web_utils import WebUtils

# Browser startup is unrelated to chat and requires a local Chrome installation.
with patch.object(WebUtils, "__init__", return_value=None):
    from app import main

REAL_ASYNC_CLIENT = httpx.AsyncClient
SEARCH = '<function_call>{"name":"web_search","parameters":{"objective":"Find official headphone weight","search_queries":["Sony WH-1000XM5 official weight"]}}</function_call>'
FETCH = '<function_call>{"name":"web_fetch","parameters":{"urls":["https://www.sony.com/headphones"]}}</function_call>'
ANSWER = "重量为 250g，来源：https://www.sony.com/headphones"


class WebChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.calls = []
        self.model_payloads = []
        self.responses = [SEARCH, FETCH, ANSWER]
        self.rpc_error = None
        self.http_status = 200
        self.sse = False
        self.wait_event = None
        self.started = asyncio.Event()
        self.api = REAL_ASYNC_CLIENT(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        )
        self.factory = patch("httpx.AsyncClient", side_effect=self.client)
        self.factory.start()

    async def asyncTearDown(self):
        self.factory.stop()
        self.env.stop()
        await self.api.aclose()

    def client(self, **kwargs):
        kwargs["transport"] = httpx.MockTransport(self.handle)
        return REAL_ASYNC_CLIENT(**kwargs)

    async def handle(self, request):
        data = json.loads(request.content)
        if str(request.url) == main.OPENAI_BASE_URL + "/chat/completions":
            self.model_payloads.append(data)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": self.responses.pop(0)}}]}
            )
        self.assertEqual(str(request.url), ENDPOINT)
        self.assertEqual(request.headers["User-Agent"], USER_AGENT)
        self.assertNotIn("authorization", request.headers)
        self.assertNotIn("x-api-key", request.headers)
        self.calls.append(data)
        method = data["method"]
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05"}
        elif method == "notifications/initialized":
            self.assertEqual(request.headers["Mcp-Session-Id"], "transport-session")
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            self.assertEqual(method, "tools/call")
            self.assertEqual(request.headers["MCP-Protocol-Version"], "2024-11-05")
            self.started.set()
            if self.wait_event:
                await self.wait_event.wait()
            if isinstance(self.rpc_error, Exception):
                raise self.rpc_error
            if self.http_status != 200:
                return httpx.Response(self.http_status, text="service unavailable")
            result = {
                "isError": bool(self.rpc_error),
                "content": [
                    {
                        "type": "text",
                        "text": "Sony weight 250g https://www.sony.com/headphones",
                    }
                ],
            }
        body = {"jsonrpc": "2.0", "id": data["id"], "result": result}
        if self.sse:
            return httpx.Response(
                200,
                text="event: message\r\ndata: " + json.dumps(body) + "\r\n\r\n",
                headers={
                    "content-type": "text/event-stream",
                    "mcp-session-id": "transport-session",
                },
            )
        return httpx.Response(
            200, json=body, headers={"mcp-session-id": "transport-session"}
        )

    async def chat(self, state=None, **kwargs):
        state = state if state is not None else {"allow_web_search": True}
        data = {
            "message": "Look up public headphone specifications",
            "history": [],
            **chat_context(state),
            **kwargs,
        }
        return await self.api.post("/chat", json=data)

    def tool_calls(self):
        return [c["params"] for c in self.calls if c["method"] == "tools/call"]

    async def test_default_anonymous_search_fetch_and_followup_consumption(self):
        state = {"allow_web_search": True}
        result = await self.chat(state)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["response"], ANSWER)
        self.assertEqual(len(result.json()["history"]), 2)
        self.assertIn(
            "https://www.sony.com/headphones", str(self.model_payloads[1]["messages"])
        )
        self.assertIn(
            "Find official headphone weight", str(self.model_payloads[2]["messages"])
        )
        self.responses = [SEARCH, ANSWER]
        self.assertEqual(
            (await self.chat(state, history=result.json()["history"])).status_code, 200
        )
        calls = self.tool_calls()
        self.assertEqual(
            [c["name"] for c in calls], ["web_search", "web_fetch", "web_search"]
        )
        self.assertEqual(
            {c["arguments"]["session_id"] for c in calls}, {state["conversation_id"]}
        )
        self.assertTrue(
            all(c["arguments"]["model_name"] == main.MIDSCENE_MODEL_NAME for c in calls)
        )
        self.assertEqual(
            calls[1]["arguments"]["urls"], ["https://www.sony.com/headphones"]
        )

    async def test_permission_off_and_saved_model_settings_preserved(self):
        self.responses = ["普通订单回复"]
        with (
            patch.object(main, "OPENAI_API_KEY", "saved-model-key"),
            patch.object(main, "MIDSCENE_MODEL_NAME", "saved-model"),
        ):
            result = await self.chat({})
        self.assertEqual(result.json()["response"], "普通订单回复")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.model_payloads[0]["model"], "saved-model")
        self.assertEqual(len(self.model_payloads[0]["messages"]), 1)

    async def test_disabled_or_unknown_explicit_provider_never_falls_back(self):
        for provider in ("none", "other-provider", ""):
            with patch.dict(os.environ, {"CHATBOT_SEARCH_PROVIDER": provider}):
                self.assertEqual((await self.chat()).status_code, 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.model_payloads, [])

    async def test_web_permission_requires_conversation_context(self):
        result = await self.api.post(
            "/chat", json={"message": "public specifications", "allow_web_search": True}
        )
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.model_payloads, [])

    async def test_explicit_parallel_and_sse(self):
        self.sse = True
        with patch.dict(
            os.environ,
            {
                "CHATBOT_SEARCH_PROVIDER": "parallel",
                "PARALLEL_API_KEY": "ambient-key-must-not-be-used",
            },
        ):
            self.assertEqual((await self.chat()).json()["response"], ANSWER)

    async def test_timeout_network_and_service_errors(self):
        for error, status in [
            (httpx.ReadTimeout("test timeout"), 200),
            (httpx.ConnectError("test network"), 200),
            (True, 200),
            (None, 429),
            (None, 500),
        ]:
            self.responses = [SEARCH]
            self.rpc_error, self.http_status = error, status
            result = await self.chat()
            self.assertEqual(result.status_code, 400)
        self.assertEqual(len(self.tool_calls()), 5)

    async def test_cancellation_propagates_from_normal_chat_caller(self):
        self.wait_event = asyncio.Event()
        task = asyncio.create_task(self.chat())
        await self.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.tool_calls()), 1)

    async def test_quota_retry_reuses_context_and_reset_changes_it(self):
        state = {"allow_web_search": True}
        self.responses, self.http_status = [SEARCH], 429
        self.assertEqual((await self.chat(state)).status_code, 400)
        old = state["conversation_id"]
        self.responses, self.http_status = [SEARCH, ANSWER], 200
        self.assertEqual((await self.chat(state)).status_code, 200)
        self.assertEqual(
            [c["arguments"]["session_id"] for c in self.tool_calls()], [old, old]
        )
        reset_chat_context(state)
        self.assertNotEqual(state["conversation_id"], old)
        self.assertFalse(state["allow_web_search"])
        self.responses = ["New chat"]
        await self.chat(state)
        self.assertEqual(len(self.tool_calls()), 2)

    async def test_interleaved_conversations_keep_distinct_ids(self):
        # Model outputs are scripted by request text, not a live model.
        async def model(message, history):
            await asyncio.sleep(0)
            return ANSWER if message.startswith("以下是网页工具") else SEARCH

        states = [{"allow_web_search": True}, {"allow_web_search": True}]
        with patch.object(main, "call_doubao_model", side_effect=model):
            results = await asyncio.gather(*(self.chat(s) for s in states))
        self.assertTrue(all(r.status_code == 200 for r in results))
        ids = {c["arguments"]["session_id"] for c in self.tool_calls()}
        self.assertEqual(ids, {s["conversation_id"] for s in states})
        self.assertEqual(len(ids), 2)
        for value in ids:
            UUID(value)

    async def test_invalid_urls_arguments_and_mixed_actions(self):
        for response in [
            FETCH.replace("https://www.sony.com/headphones", "file:///etc/passwd"),
            SEARCH.replace(
                '"search_queries":["Sony WH-1000XM5 official weight"]',
                '"search_queries":[]',
            ),
            SEARCH
            + '<function_call>{"name":"open_url","parameters":{}}</function_call>',
        ]:
            self.responses = [response]
            self.assertEqual((await self.chat()).status_code, 400)
        self.assertEqual(self.calls, [])

    async def test_web_results_cannot_trigger_order_actions(self):
        action = '<function_call>{"name":"open_url","parameters":{"url":"https://example.com"}}</function_call>'
        self.responses = [SEARCH, action]
        self.assertEqual((await self.chat()).status_code, 400)

    async def test_bounded_loop_and_existing_order_dispatch(self):
        self.responses = [SEARCH] * 4
        self.assertEqual((await self.chat()).status_code, 400)
        self.assertEqual(len(self.tool_calls()), 3)
        self.calls.clear()
        action = (
            '<function_call>{"name":"take_screenshot","parameters":{}}</function_call>'
        )
        self.responses = [action]
        self.assertEqual((await self.chat()).json()["response"], action)
        self.assertEqual(self.calls, [])


class FrontendContextTests(unittest.TestCase):
    def test_streamlit_permission_and_clear_lifecycle(self):
        import sys
        from streamlit.testing.v1 import AppTest

        frontend = Path(__file__).parents[1] / "app/frontend.py"
        with patch.object(sys, "path", [str(frontend.parent)] + sys.path):
            ui = AppTest.from_file(str(frontend), default_timeout=15).run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.checkbox(key="allow_web_search").value)
            ui.checkbox(key="allow_web_search").check().run()
            old = chat_context(ui.session_state)["conversation_id"]
            ui.button(key="clear_chat").click().run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.checkbox(key="allow_web_search").value)
            self.assertNotEqual(chat_context(ui.session_state)["conversation_id"], old)

    def test_real_frontend_wires_both_requests_and_clear(self):
        tree = ast.parse((Path(__file__).parents[1] / "app/frontend.py").read_text())
        requests = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "chat_request" for t in n.targets
            )
        ]
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertTrue(
                any(
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name)
                    and v.func.id == "chat_context"
                    for v in request.value.values
                )
            )
        clear = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "clear_chat_history"
        )
        self.assertTrue(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "reset_chat_context"
                for n in ast.walk(clear)
            )
        )


if __name__ == "__main__":
    unittest.main()
