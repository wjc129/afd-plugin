# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""OpenAI-compatible proxy for native Prefill plus AFD Decode serving."""

from __future__ import annotations

import argparse
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

DEFAULT_PROXY_HOST = "0.0.0.0"
DEFAULT_PROXY_PORT = 8000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1800
PREFILL_MAX_TOKENS = 1
COMPLETIONS_ENDPOINT = "/v1/completions"
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route each request through native Prefill and AFD Decode",
    )
    parser.add_argument("--host", default=DEFAULT_PROXY_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


def create_app(
    *,
    prefill_url: str,
    decode_url: str,
    request_timeout: int,
) -> FastAPI:
    prefill_url = prefill_url.rstrip("/")
    decode_url = decode_url.rstrip("/")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        app.state.http_session = aiohttp.ClientSession(timeout=timeout)
        try:
            yield
        finally:
            await app.state.http_session.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        session: aiohttp.ClientSession = request.app.state.http_session
        headers = {}
        authorization = request.headers.get("authorization")
        if authorization is not None:
            headers["authorization"] = authorization
        async with session.get(
            f"{decode_url}/v1/models",
            headers=headers,
        ) as decode_response:
            body = await decode_response.read()
            return Response(
                content=body,
                status_code=decode_response.status,
                media_type=decode_response.headers.get("content-type"),
            )

    async def route_openai_request(request: Request, endpoint: str) -> Response:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="request body must be an object"
            )

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        headers = {"x-request-id": request_id}
        authorization = request.headers.get("authorization")
        if authorization is not None:
            headers["authorization"] = authorization
        session: aiohttp.ClientSession = request.app.state.http_session

        prefill_payload = dict(payload)
        prefill_payload["stream"] = False
        if "max_completion_tokens" in prefill_payload:
            prefill_payload["max_completion_tokens"] = PREFILL_MAX_TOKENS
            prefill_payload.pop("max_tokens", None)
        else:
            prefill_payload["max_tokens"] = PREFILL_MAX_TOKENS
        prefill_payload.pop("stream_options", None)
        prefill_payload["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }

        async with session.post(
            f"{prefill_url}{endpoint}",
            json=prefill_payload,
            headers=headers,
        ) as prefill_response:
            prefill_body = await prefill_response.read()
            if prefill_response.status != 200:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Prefill service failed: "
                        f"HTTP {prefill_response.status} "
                        f"{prefill_body.decode(errors='replace')}"
                    ),
                )
            prefill_result = await prefill_response.json()

        kv_transfer_params = prefill_result.get("kv_transfer_params")
        if not isinstance(kv_transfer_params, dict) or not kv_transfer_params:
            raise HTTPException(
                status_code=502,
                detail="Prefill response did not contain kv_transfer_params",
            )

        decode_payload = dict(payload)
        decode_payload["kv_transfer_params"] = kv_transfer_params
        decode_response = await session.post(
            f"{decode_url}{endpoint}",
            json=decode_payload,
            headers=headers,
        )
        if decode_response.status != 200:
            decode_body = await decode_response.read()
            decode_response.release()
            raise HTTPException(
                status_code=502,
                detail=(
                    "Decode service failed: "
                    f"HTTP {decode_response.status} "
                    f"{decode_body.decode(errors='replace')}"
                ),
            )

        media_type = decode_response.headers.get("content-type")
        if bool(payload.get("stream", False)):

            async def stream_decode_body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in decode_response.content.iter_any():
                        yield chunk
                finally:
                    decode_response.release()

            return StreamingResponse(
                stream_decode_body(),
                status_code=decode_response.status,
                media_type=media_type,
            )

        decode_body = await decode_response.read()
        decode_response.release()
        return Response(
            content=decode_body,
            status_code=200,
            media_type=media_type,
        )

    @app.post(COMPLETIONS_ENDPOINT)
    async def completions(request: Request) -> Response:
        return await route_openai_request(request, COMPLETIONS_ENDPOINT)

    @app.post(CHAT_COMPLETIONS_ENDPOINT)
    async def chat_completions(request: Request) -> Response:
        return await route_openai_request(request, CHAT_COMPLETIONS_ENDPOINT)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
        request_timeout=args.request_timeout,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
