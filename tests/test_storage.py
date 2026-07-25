import os
import pickle
import tempfile
from pathlib import Path

import pytest

from kel.runtime.checkpoint import Checkpoint
from kel.storage import ArtifactStore, FileCheckpointStore, LocalBlobStore
from kel.storage.s3 import S3BlobStore


def test_local_blob_store_put_get_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalBlobStore(tmp)
        content_id = store.put(b"hello world")
        assert store.get(content_id) == b"hello world"
        assert store.exists(content_id)


def test_local_blob_store_is_content_addressed_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalBlobStore(tmp)
        id1 = store.put(b"same content")
        id2 = store.put(b"same content")
        assert id1 == id2
        # only one file should exist under that content id's shard
        shard_dir = Path(tmp) / id1[:2]
        assert len(list(shard_dir.iterdir())) == 1


def test_local_blob_store_get_missing_raises():
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalBlobStore(tmp)
        with pytest.raises(KeyError):
            store.get("deadbeef")


def test_local_blob_store_delete():
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalBlobStore(tmp)
        content_id = store.put(b"transient")
        store.delete(content_id)
        assert not store.exists(content_id)


def test_artifact_store_save_load_by_name_and_persists_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        blob_dir = Path(tmp) / "blobs"
        manifest_dir = Path(tmp) / "artifacts"
        store = ArtifactStore(LocalBlobStore(blob_dir), manifest_dir)
        store.save("trace-export", b'{"spans": []}', content_type="application/json")

        assert store.load("trace-export") == b'{"spans": []}'
        assert store.content_type("trace-export") == "application/json"
        assert store.list() == ["trace-export"]

        # manifest survives a fresh ArtifactStore instance pointed at the same directory
        reloaded = ArtifactStore(LocalBlobStore(blob_dir), manifest_dir)
        assert reloaded.load("trace-export") == b'{"spans": []}'


def test_artifact_store_load_missing_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(LocalBlobStore(tmp), tmp)
        with pytest.raises(KeyError):
            store.load("nope")


def test_file_checkpoint_store_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        store1 = FileCheckpointStore(tmp)
        store1.save(Checkpoint(run_id="run-1", step=1, node="a", state={"x": 1}))
        store1.save(Checkpoint(run_id="run-1", step=2, node="b", state={"x": 2}))

        store2 = FileCheckpointStore(tmp)
        history = store2.history("run-1")
        assert [c.node for c in history] == ["a", "b"]
        assert store2.load_latest("run-1").state == {"x": 2}


def test_file_checkpoint_store_empty_run_returns_empty_and_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        assert store.history("missing") == []
        assert store.load_latest("missing") is None


class _EvilPayload:
    """Classic pickle-RCE proof-of-concept: __reduce__ tells the unpickler
    to call os.system(...) during load. This is the exact mechanism behind
    deserialization-of-untrusted-data CVEs like LangChain's CVE-2025-68664."""

    def __reduce__(self):
        return (os.system, ("echo pwned > pwned.txt",))


def test_file_checkpoint_store_refuses_malicious_pickle_payload():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        # simulate a tampered/malicious checkpoint file written directly to disk,
        # bypassing store.save() the way an attacker with file write access would
        store._path("evil-run").write_bytes(pickle.dumps([_EvilPayload()]))

        with pytest.raises(pickle.UnpicklingError, match="refusing to unpickle"):
            store.history("evil-run")

        # and to be thorough: confirm the payload never actually ran
        assert not (Path(tmp) / "pwned.txt").exists()
        assert not Path("pwned.txt").exists()


def test_file_checkpoint_store_allows_legitimate_kel_types_including_pydantic():
    # sanity check that the allowlist isn't so tight it breaks real usage —
    # Checkpoint.state holding a pydantic Message must still round-trip
    from kel.models.types import Message

    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        store.save(Checkpoint(run_id="run-2", step=1, node="a", state={"msg": Message.user("hi")}))

        loaded = store.load_latest("run-2")
        assert loaded.state["msg"].text == "hi"


class _FakeS3Client:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


def test_s3_blob_store_put_get_exists_delete_with_fake_client():
    client = _FakeS3Client()
    store = S3BlobStore("my-bucket", client=client)

    content_id = store.put(b"s3 content")
    assert store.exists(content_id)
    assert store.get(content_id) == b"s3 content"

    store.delete(content_id)
    assert not store.exists(content_id)


def test_s3_blob_store_put_is_idempotent_and_avoids_reupload():
    client = _FakeS3Client()
    store = S3BlobStore("my-bucket", client=client)

    id1 = store.put(b"same")
    id2 = store.put(b"same")
    assert id1 == id2
    assert len(client.objects) == 1


def test_s3_blob_store_construction_never_requires_explicit_credentials():
    """Contract test: S3BlobStore(bucket, client=...) must not require any
    api_key/secret/token kwarg — this is what makes it work unmodified on
    EC2 (instance profile role), EKS (IRSA), and ECS (task role), where
    boto3's own default credential chain resolves creds with nothing
    explicit passed. Passing only `bucket` and `client` must be sufficient."""
    import inspect

    sig = inspect.signature(S3BlobStore.__init__)
    required_params = [
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD
    ]
    assert required_params == ["bucket"]  # the only required arg; no credential param is required

    # and it actually constructs fine with just bucket + an injected client
    store = S3BlobStore("my-bucket", client=_FakeS3Client())
    assert store.bucket == "my-bucket"
