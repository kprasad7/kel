"""S3-compatible BlobStore backend. Requires boto3 (`pip install kel[s3]`)
— kept out of core the same way provider SDKs are, so the zero-dependency
LocalBlobStore stays zero-dependency.

**No credentials are ever forced here.** `S3BlobStore("my-bucket")` with
no other arguments calls `boto3.client("s3")` with nothing explicit, which
means boto3's own default credential chain resolves them — an EC2 instance
profile role, an EKS pod's IRSA-injected role (via the
`AWS_WEB_IDENTITY_TOKEN_FILE`/`AWS_ROLE_ARN` env vars Kubernetes sets
automatically), an ECS task role, or `~/.aws/credentials` — all without
kel needing to know which one applies. Pass explicit `aws_access_key_id`/
`aws_secret_access_key` via `**client_kwargs` only if you actually need to
override that chain (e.g. local dev against a non-default profile)."""

from __future__ import annotations

import hashlib
from typing import Any


class S3BlobStore:
    def __init__(self, bucket: str, *, prefix: str = "", client: Any = None, **client_kwargs: Any):
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "S3BlobStore requires boto3. Install it with `pip install kel[s3]`."
                ) from exc
            self._client = boto3.client("s3", **client_kwargs)
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, content_id: str) -> str:
        return f"{self.prefix}{content_id[:2]}/{content_id}"

    def put(self, data: bytes) -> str:
        content_id = hashlib.sha256(data).hexdigest()
        key = self._key(content_id)
        if not self.exists(content_id):
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return content_id

    def get(self, content_id: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=self._key(content_id))
        return obj["Body"].read()

    def exists(self, content_id: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(content_id))
            return True
        except Exception:
            return False

    def delete(self, content_id: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self._key(content_id))
