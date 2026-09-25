"""Secret-shaped strings are found, named by kind only, and kept out of
the tree: tools/secret_scan.py (CI runs it over every tracked file),
tools/pull_box_records.py (withholds a pulled record that has one),
scripts/rent_box.py's scrubber. Fixture values are built at run time, so
this file holds none."""
import sys
from types import SimpleNamespace

import pytest

import scripts.rent_box as rent_box
from tools import pull_box_records, secret_scan

ALNUM = "Xy7" * 14                                  # 42 characters of token alphabet, not hex
HEX64 = "0123456789abcdef" * 4


def _fixtures():
    return {
        "huggingface token": "token read: " + "hf" + "_" + ALNUM,
        "api_key parameter": "GET https://example.invalid/api/v0/asks/1/?" + "api" + "_key=" + ALNUM,
        "bearer credential": "Illegal header value b'" + "Bear" + "er " + ALNUM + "'",
        "container api key": "CONTAINER" + "_API_KEY=" + ALNUM,
        "hex key": "api" + "_key: " + HEX64,
        "private key": "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
        "kaggle token": "export KAGGLE" + "_API_TOKEN=" + ALNUM,
    }


def test_each_kind_is_found_and_named_without_its_value():
    for kind, line in _fixtures().items():
        hits = secret_scan.scan_text("a quiet line\n" + line + "\n")
        assert hits == [(2, kind)], (kind, hits)
    assert secret_scan.scan_text("hf_short and Bearer <redacted> and api_key=<redacted>") == []
    marked = "hf" + "_" + ALNUM + "  # " + secret_scan.ALLOW_MARKER
    assert secret_scan.scan_text(marked) == []


def test_binary_and_oversized_files_are_not_read(tmp_path):
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\0\1" + ("hf" + "_" + ALNUM).encode())
    assert secret_scan.scan_file(binary) == []
    text = tmp_path / "log.txt"
    text.write_text(_fixtures()["huggingface token"], encoding="utf-8")
    assert secret_scan.scan_file(text) == [(1, "huggingface token")]


def test_a_pulled_record_with_a_secret_is_withheld(tmp_path, monkeypatch):
    host = tmp_path / "host"
    host.mkdir()
    (host / "train.log").write_text("step 1 loss 2.0\n", encoding="utf-8")
    (host / "stop.log").write_text("stop failed: " + _fixtures()["api_key parameter"] + "\n",
                                   encoding="utf-8")
    files = {"tier-b/run/train.log": host / "train.log", "tier-b/run/stop.log": host / "stop.log"}

    class FakeApi:
        def list_repo_tree(self, repo, path_in_repo):
            return [SimpleNamespace(path=p, size=f.stat().st_size) for p, f in files.items()]

    hub = SimpleNamespace(HfApi=FakeApi, hf_hub_download=lambda repo, path: str(files[path]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    dest = tmp_path / "records"
    assert pull_box_records.pull("tier-b/run", dest) == 1
    assert (dest / "train.log").exists() and not (dest / "stop.log").exists()


@pytest.mark.parametrize("line", ["Authorization: " + "Bear" + "er " + ALNUM,
                                  "CONTAINER" + "_API_KEY=" + ALNUM])
def test_the_rent_tool_scrubs_bearer_and_container_keys(line):
    scrubbed = rent_box.scrub_text(line)
    assert ALNUM not in scrubbed and rent_box.REDACTED in scrubbed
