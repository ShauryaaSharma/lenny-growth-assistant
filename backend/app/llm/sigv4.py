"""AWS Signature Version 4, as an httpx auth hook.

Bedrock accepts either a Bedrock API key (a bearer token) or a request signed
with IAM credentials. Signing is ~50 lines of documented HMAC chaining, so it is
written here rather than pulling in botocore for one function -- the providers in
this package all speak plain httpx. `tests/test_cloud_providers.py` checks it
against the worked example in AWS's own SigV4 documentation.

Credentials come from the standard environment variables (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, optional AWS_SESSION_TOKEN). Shared-config profiles and
instance-metadata roles are not resolved: export their credentials, or use a
Bedrock API key.

Reference: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from urllib.parse import quote, unquote

import httpx

ALGORITHM = "AWS4-HMAC-SHA256"


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(value: str) -> str:
    return quote(value, safe="-_.~")


def canonical_query(raw: str) -> str:
    pairs = []
    for part in filter(None, raw.split("&")):
        key, _, value = part.partition("=")
        pairs.append((_encode(unquote(key)), _encode(unquote(value))))
    return "&".join(f"{k}={v}" for k, v in sorted(pairs))


class SigV4Auth(httpx.Auth):
    requires_request_body = True

    def __init__(self, access_key: str, secret_key: str, region: str, service: str,
                 session_token: str = "",
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.service = service
        self.session_token = session_token
        self.clock = clock

    def sign(self, request: httpx.Request) -> None:
        amz_date = request.headers.get("x-amz-date") or self.clock().strftime("%Y%m%dT%H%M%SZ")
        request.headers["x-amz-date"] = amz_date
        if self.session_token:
            request.headers["x-amz-security-token"] = self.session_token

        signed = {k: " ".join(request.headers[k].split())
                  for k in ("content-type", "host", "x-amz-date", "x-amz-security-token")
                  if k in request.headers}
        signed_names = ";".join(sorted(signed))
        # Non-S3 services take the already-encoded path encoded once more, so a
        # Bedrock model id's ":" -- sent as %3A -- is signed as %253A.
        path = request.url.raw_path.decode().split("?", 1)[0] or "/"
        canonical = "\n".join([
            request.method,
            quote(path, safe="/-_.~"),
            canonical_query(request.url.query.decode()),
            "".join(f"{k}:{signed[k]}\n" for k in sorted(signed)),
            signed_names,
            _sha256(request.content),
        ])

        day = amz_date[:8]
        scope = f"{day}/{self.region}/{self.service}/aws4_request"
        to_sign = "\n".join([ALGORITHM, amz_date, scope, _sha256(canonical.encode())])
        key = _hmac(_hmac(_hmac(_hmac(f"AWS4{self.secret_key}".encode(), day),
                                self.region), self.service), "aws4_request")
        signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
        request.headers["authorization"] = (
            f"{ALGORITHM} Credential={self.access_key}/{scope}, "
            f"SignedHeaders={signed_names}, Signature={signature}"
        )

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        self.sign(request)
        yield request
