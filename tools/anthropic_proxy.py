"""
Anthropic Messages API -> vLLM OpenAI API proxy.

Expose:
  POST /v1/messages
  GET  /v1/models
  GET  /health

Forward to upstream vLLM OpenAI API (configurable via UPSTREAM_BASE_URL).
"""
import os
import json
import uuid
from typing import AsyncIterator, Any

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "http://127.0.0.1:8000/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "sk-vllm")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")

app = FastAPI(title="Anthropic-to-OpenAI Proxy")

http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = 4096
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stream: bool | None = False
    stop_sequences: list[str] | None = Field(default=None, alias="stop_sequences")
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    thinking: dict[str, Any] | None = None


def _require_auth(request: Request) -> None:
    if not PROXY_API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _anthropic_role_to_openai(role: str) -> str:
    if role == "user":
        return "user"
    if role in ("assistant",):
        return "assistant"
    raise ValueError(f"Unsupported anthropic role: {role}")


def _normalize_content(content: str | list[dict[str, Any]] | None) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Convert anthropic content blocks to OpenAI content parts
    parts = []
    for block in content:
        t = block.get("type")
        if t == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif t == "tool_use":
            # OpenAI tool_calls representation in assistant message
            # This is handled separately in message conversion
            continue
        elif t == "tool_result":
            # tool_result becomes a user message with tool_call_id
            continue
        elif t == "thinking":
            # Drop internal thinking blocks for now
            continue
        elif t == "redacted_thinking":
            continue
        else:
            parts.append({"type": "text", "text": json.dumps(block)})
    return parts if parts else ""


def _convert_messages(anthropic_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic messages to OpenAI messages.

    Anthropic puts tool_use blocks inside assistant message content and tool_result
    inside user message content. OpenAI uses tool_calls/tool_call_id fields.
    """
    openai_messages = []

    # Handle system separately; for now we fold it into a system message if provided
    # by caller via 'system' top-level param. Messages with role=system are rare in
    # Anthropic but keep them if present.
    for msg in anthropic_messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            openai_messages.append({"role": "system", "content": content if isinstance(content, str) else json.dumps(content)})
            continue

        if role == "user":
            if isinstance(content, list):
                text_parts = []
                tool_results = []
                for block in content:
                    t = block.get("type")
                    if t == "text":
                        text_parts.append(block.get("text", ""))
                    elif t == "tool_result":
                        tool_call_id = block.get("tool_use_id", "")
                        tool_content = ""
                        tc = block.get("content")
                        if isinstance(tc, str):
                            tool_content = tc
                        elif isinstance(tc, list):
                            tool_content = "\n".join(
                                x.get("text", "") for x in tc if x.get("type") == "text"
                            )
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_content,
                        })
                # Anthropic tool_result blocks live in the same user turn; OpenAI
                # splits them into separate tool messages. We put text first, then tools.
                if text_parts:
                    openai_messages.append({"role": "user", "content": "\n".join(text_parts)})
                openai_messages.extend(tool_results)
            else:
                openai_messages.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    t = block.get("type")
                    if t == "text":
                        text_parts.append(block.get("text", ""))
                    elif t == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = ""
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_messages.append(assistant_msg)
            else:
                openai_messages.append({"role": "assistant", "content": content})
            continue

        raise ValueError(f"Unsupported message role: {role}")

    return openai_messages


def _build_system_message(system: str | list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if system is None:
        return None
    if isinstance(system, str):
        return {"role": "system", "content": system}
    # List of text blocks
    texts = []
    for block in system:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
    if texts:
        return {"role": "system", "content": "\n".join(texts)}
    return None


def _convert_tool_choice(tool_choice: dict[str, Any] | str | None) -> dict[str, Any] | str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        if tool_choice == "any":
            return "auto"  # OpenAI has no exact "any"; map to auto
        return None
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "auto":
            return "auto"
        if t == "none":
            return "none"
        if t == "tool":
            name = tool_choice.get("name", "")
            return {"type": "function", "function": {"name": name}}
        if t == "any":
            return "auto"
    return None


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    openai_tools = []
    for tool in tools:
        if tool.get("type") == "custom" or "custom" in tool:
            # Anthropic custom tool wrapper; unwrap input_schema
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        elif tool.get("type") == "function" or "function" in tool:
            # Already OpenAI-like
            openai_tools.append(tool)
        else:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return openai_tools


def _openai_to_anthropic(openai_resp: dict[str, Any]) -> dict[str, Any]:
    choice = openai_resp["choices"][0]
    msg = choice.get("message", {})
    content = []

    # OpenAI reasoning_content (e.g. Qwen3 thinking) -> Anthropic thinking block
    reasoning = msg.get("reasoning_content") or ""
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    text = msg.get("content") or ""
    if text:
        content.append({"type": "text", "text": text})

    for tc in msg.get("tool_calls", []):
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc["function"]["name"],
            "input": args,
        })

    stop_reason = choice.get("finish_reason")
    anthropic_stop_reason = None
    if stop_reason == "stop":
        anthropic_stop_reason = "end_turn"
    elif stop_reason == "length":
        anthropic_stop_reason = "max_tokens"
    elif stop_reason == "tool_calls":
        anthropic_stop_reason = "tool_use"

    usage = openai_resp.get("usage", {})
    return {
        "id": f"msg_{openai_resp.get('id', str(uuid.uuid4()).replace('-', ''))[:24]}",
        "type": "message",
        "role": "assistant",
        "model": openai_resp.get("model", ""),
        "content": content,
        "stop_reason": anthropic_stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def _stream_openai_to_anthropic(openai_stream: httpx.Response) -> AsyncIterator[str]:
    """Convert OpenAI SSE stream to Anthropic SSE stream.

    Anthropic streaming events:
      event: message_start
      data: {"type":"message_start","message":{...}}

      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}

      event: content_block_stop
      data: {"type":"content_block_stop","index":0}

      event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":123}}

      event: message_stop
      data: {"type":"message_stop"}
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = ""
    input_tokens = 0
    index = 0
    started = False

    # Send message_start skeleton; real model/usage filled at end
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':message_id,'type':'message','role':'assistant','model':model_name,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"

    async for line in openai_stream.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if not model_name:
                model_name = chunk.get("model", "")

            delta = chunk["choices"][0].get("delta", {})
            finish_reason = chunk["choices"][0].get("finish_reason")

            # Reasoning/thinking delta
            reasoning_delta = delta.get("reasoning_content") or ""
            if reasoning_delta:
                if not started:
                    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':index,'content_block':{'type':'thinking','thinking':''}})}\n\n"
                    started = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':index,'delta':{'type':'thinking_delta','thinking':reasoning_delta}})}\n\n"

            # Text delta
            text_delta = delta.get("content") or ""
            if text_delta:
                if not started:
                    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':index,'content_block':{'type':'text','text':''}})}\n\n"
                    started = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':index,'delta':{'type':'text_delta','text':text_delta}})}\n\n"

            # Tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                if not started:
                    # Start tool_use block
                    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':index,'content_block':{'type':'tool_use','id':tc_delta.get('id',''),'name':tc_delta['function'].get('name',''),'input':{}}})}\n\n"
                    started = True
                arg_delta = tc_delta["function"].get("arguments", "")
                if arg_delta:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':index,'delta':{'type':'input_json_delta','partial_json':arg_delta}})}\n\n"

            usage = chunk.get("usage", {})
            if usage.get("prompt_tokens"):
                input_tokens = usage["prompt_tokens"]

            if finish_reason:
                if started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':index})}\n\n"
                stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
                anthropic_stop = stop_reason_map.get(finish_reason)
                yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':anthropic_stop,'stop_sequence':None},'usage':{'output_tokens':usage.get('completion_tokens',0)}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"

    if started and not finish_reason:
        # Stream ended without finish_reason; close block anyway
        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':index})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request):
    _require_auth(request)
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}"}
    resp = await http_client.get(f"{UPSTREAM_BASE_URL}/models", headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))


@app.post("/v1/messages")
async def create_message(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
        req = AnthropicMessageRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    openai_messages = _convert_messages(req.messages)
    system_msg = _build_system_message(req.system)
    if system_msg:
        openai_messages.insert(0, system_msg)

    openai_body: dict[str, Any] = {
        "model": req.model,
        "messages": openai_messages,
        "max_tokens": req.max_tokens or 4096,
        "stream": req.stream or False,
    }
    if req.temperature is not None:
        openai_body["temperature"] = req.temperature
    if req.top_p is not None:
        openai_body["top_p"] = req.top_p
    if req.stop_sequences:
        openai_body["stop"] = req.stop_sequences

    tools = _convert_tools(req.tools)
    if tools:
        openai_body["tools"] = tools
        tc = _convert_tool_choice(req.tool_choice)
        if tc:
            openai_body["tool_choice"] = tc

    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json",
    }

    if openai_body["stream"]:
        upstream_resp = await http_client.post(
            f"{UPSTREAM_BASE_URL}/chat/completions",
            headers=headers,
            json=openai_body,
            timeout=httpx.Timeout(600.0),
        )
        if upstream_resp.status_code != 200:
            return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, media_type="application/json")
        return StreamingResponse(
            _stream_openai_to_anthropic(upstream_resp),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    upstream_resp = await http_client.post(
        f"{UPSTREAM_BASE_URL}/chat/completions",
        headers=headers,
        json=openai_body,
        timeout=httpx.Timeout(600.0),
    )

    if upstream_resp.status_code != 200:
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, media_type="application/json")

    openai_data = upstream_resp.json()
    anthropic_data = _openai_to_anthropic(openai_data)
    return JSONResponse(content=anthropic_data)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "18081"))
    uvicorn.run(app, host=host, port=port)
