from __future__ import annotations

import base64
import hashlib
import io
import ssl
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import download_weights


class FakeSSLContext:
    def __init__(self, roots=()):
        self.verify_mode = ssl.CERT_REQUIRED
        self.check_hostname = True
        self.roots = list(roots)

    def load_verify_locations(self, *, cafile=None, capath=None, cadata=None):
        self.roots.append((cafile, capath, cadata))


def test_existing_default_cafile_leaves_platform_context_untouched(monkeypatch, tmp_path):
    cafile = tmp_path / "system-ca.pem"
    cafile.write_text("system trust marker")
    context = FakeSSLContext(("system capath roots",))
    created = []

    def create_default_context(*args, **kwargs):
        created.append((args, kwargs))
        return context

    monkeypatch.setattr(ssl, "create_default_context", create_default_context)
    monkeypatch.setattr(ssl, "get_default_verify_paths",
                        lambda: SimpleNamespace(cafile=str(cafile), capath="/system/certs"))

    result = download_weights._verified_ssl_context()

    assert result is context
    assert created == [((), {})]
    assert context.roots == ["system capath roots"]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_missing_default_cafile_adds_certifi_without_replacing_other_roots(
        monkeypatch, tmp_path):
    context = FakeSSLContext(("system capath roots", "custom capath roots"))
    certifi_bundle = tmp_path / "certifi.pem"
    certifi_bundle.write_text("certifi trust marker")
    order = []

    def create_default_context(*args, **kwargs):
        order.append("create")
        return context

    monkeypatch.setattr(ssl, "create_default_context", create_default_context)
    monkeypatch.setattr(ssl, "get_default_verify_paths",
                        lambda: SimpleNamespace(cafile=str(tmp_path / "missing.pem"),
                                                capath="/custom/certs"))
    monkeypatch.setattr(download_weights.certifi, "where", lambda: str(certifi_bundle))

    result = download_weights._verified_ssl_context()

    assert result is context
    assert context.roots == ["system capath roots", "custom capath roots",
                             (str(certifi_bundle), None, None)]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert order == ["create"]


@pytest.mark.parametrize("cafile", [None, ""])
def test_absent_default_cafile_adds_certifi_bundle(monkeypatch, tmp_path, cafile):
    context = FakeSSLContext()
    bundle = tmp_path / "certifi.pem"
    bundle.touch()
    monkeypatch.setattr(ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(ssl, "get_default_verify_paths",
                        lambda: SimpleNamespace(cafile=cafile, capath=None))
    monkeypatch.setattr(download_weights.certifi, "where", lambda: str(bundle))

    assert download_weights._verified_ssl_context() is context
    assert context.roots == [(str(bundle), None, None)]
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_rvm_download_passes_verified_context_and_still_checks_hash(monkeypatch, tmp_path):
    payload = b"offline synthetic checkpoint"
    digest = base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")
    context = FakeSSLContext(("system roots",))
    calls = []
    monkeypatch.setattr(download_weights, "_verified_ssl_context", lambda: context)
    monkeypatch.setattr(download_weights, "RVM_MD5_B64", digest)

    def offline_urlopen(url, *, context):
        calls.append((url, context))
        return io.BytesIO(payload)

    monkeypatch.setattr(download_weights.urllib.request, "urlopen", offline_urlopen)
    root = tmp_path / "package"

    download_weights.download_rvm(root)

    destination = root / "vfms/rvm" / download_weights.RVM_FILE
    assert destination.read_bytes() == payload
    assert calls == [(download_weights.RVM_URL, context)]
    assert context.verify_mode == ssl.CERT_REQUIRED

    monkeypatch.setattr(download_weights, "RVM_MD5_B64", "not-the-checkpoint-hash")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_weights.download_rvm(root)
    assert destination.with_suffix(destination.suffix + ".part").exists() is False
