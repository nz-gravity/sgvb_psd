"""Release requests must be deliberate and must not replace other versions."""

import importlib.util
import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/release.py"
spec = importlib.util.spec_from_file_location("release_workflow", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
MAIN = "refs/heads/main"


@pytest.mark.parametrize(
    "message",
    [
        "feat!: replace the backend",
        "Merge pull request #13\n\nrelease: v2.1.0",
        "docs: explain release: v2.1.0",
        "",
    ],
)
def test_ordinary_commits_never_publish(message):
    assert (
        release.requested_version(
            "push", MAIN, {"head_commit": {"message": message}}
        )
        is None
    )


@pytest.mark.parametrize("version", ["2.0.1", "2.1.0", "2.1.0rc1", "2.1.0b2"])
def test_exact_subject_and_manual_requests(version):
    event = {"head_commit": {"message": f"release: v{version}\n\nNotes"}}
    assert release.requested_version("push", MAIN, event) == version
    assert (
        release.requested_version(
            "workflow_dispatch", MAIN, {"inputs": {"version": version}}
        )
        == version
    )


@pytest.mark.parametrize(
    "version",
    [
        "2.1",
        "02.1.0",
        "2.1.0+local",
        "2.1.0.dev1",
        "2.1.0.post1",
        "1!2.1.0",
        "",
        "$(whoami)",
    ],
)
def test_invalid_versions_fail(version):
    with pytest.raises(ValueError):
        release.validate_version(version)


@pytest.mark.parametrize(
    "message",
    [
        "release: 2.1.0",
        "release: v2.1.0 extra",
        "release: v2.1.0; echo unsafe",
    ],
)
def test_malformed_release_subject_fails(message):
    with pytest.raises(ValueError):
        release.requested_version(
            "push", MAIN, {"head_commit": {"message": message}}
        )


def test_manual_release_cannot_publish_a_feature_branch():
    with pytest.raises(ValueError, match="only allowed from main"):
        release.requested_version(
            "workflow_dispatch",
            "refs/heads/feature/jax-sgvb",
            {"inputs": {"version": "2.1.0"}},
        )


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release.git("init", "--quiet")
    release.git("config", "user.email", "test@example.org")
    release.git("config", "user.name", "Release test")
    release.git("config", "commit.gpgsign", "false")
    release.git("commit", "--quiet", "--allow-empty", "-m", "initial")
    release.git("tag", "v2.0.0")
    release.git("commit", "--quiet", "--allow-empty", "-m", "updated source")
    return release.git("rev-parse", "HEAD")


def test_new_version_and_same_commit_retry(repository):
    release.validate_tags("2.1.0rc1", repository)
    release.git("tag", "v2.1.0rc1")
    release.validate_tags("2.1.0rc1", repository)
    release.validate_tags("2.1.0", repository)


def test_existing_version_on_another_commit_is_rejected(repository):
    with pytest.raises(ValueError, match="different commit"):
        release.validate_tags("2.0.0", repository)


def test_semantic_version_ordering_prevents_older_releases(repository):
    release.git("tag", "v2.10.0")
    with pytest.raises(ValueError, match="newer than 2.10.0"):
        release.validate_tags("2.9.0", repository)


def test_cli_writes_only_validated_outputs(repository, tmp_path):
    event = tmp_path / "event.json"
    output = tmp_path / "output"
    event.write_text(
        json.dumps({"head_commit": {"message": "release: v2.1.0"}})
    )
    env = {
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": MAIN,
        "GITHUB_SHA": repository,
        "GITHUB_OUTPUT": str(output),
    }
    subprocess.run(
        [sys.executable, str(SCRIPT)], env={**os.environ, **env}, check=True
    )
    fields = dict(
        line.split("=", 1) for line in output.read_text().splitlines()
    )
    assert fields == {
        "publish": "true",
        "version": "2.1.0",
        "tag": "v2.1.0",
        "sha": repository,
        "prerelease": "false",
    }


def test_github_squash_subject_suffix_is_accepted():
    event = {"head_commit": {"message": "release: v2.1.0 (#14)"}}
    assert release.requested_version("push", MAIN, event) == "2.1.0"


def test_cli_does_not_emit_a_request_after_validation_failure(
    repository, tmp_path, monkeypatch
):
    event = tmp_path / "event.json"
    output = tmp_path / "output"
    event.write_text(json.dumps({"inputs": {"version": "2.0.0"}}))
    for name, value in {
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": MAIN,
        "GITHUB_SHA": repository,
        "GITHUB_OUTPUT": str(output),
    }.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="different commit"):
        release.main()
    assert not output.exists()
