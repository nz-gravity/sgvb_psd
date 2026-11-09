# /// script
# requires-python = ">=3.10"
# dependencies = ["packaging==25.0"]
# ///
"""Validate explicit release requests; never infer versions from history."""

import json
import os
import re
import subprocess
from pathlib import Path

from packaging.version import InvalidVersion, Version


def requested_version(event_name, ref, event):
    if ref != "refs/heads/main":
        raise ValueError("Releases are only allowed from main")
    if event_name == "workflow_dispatch":
        value = event.get("inputs", {}).get("version", "")
        return validate_version(value.removeprefix("v"))
    if event_name != "push":
        raise ValueError(f"Unsupported release event: {event_name}")
    message = (event.get("head_commit") or {}).get("message", "")
    subject = message.splitlines()[0] if message else ""
    if not subject.startswith("release:"):
        return None
    match = re.fullmatch(r"release: v(\S+)(?: \(#[1-9]\d*\))?", subject)
    if not match:
        raise ValueError("Use the exact commit subject: release: v2.1.0")
    return validate_version(match[1])


def validate_version(value):
    try:
        version = Version(value)
    except InvalidVersion as error:
        raise ValueError("Invalid release version") from error
    if (
        str(version) != value
        or len(version.release) != 3
        or version.epoch
        or version.local
        or version.is_devrelease
        or version.is_postrelease
    ):
        raise ValueError("Use X.Y.Z or a canonical prerelease, e.g. 2.1.0rc1")
    return value


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_tags(version, sha):
    tags = git("tag", "--list").splitlines()
    tag = f"v{version}"
    if tag in tags and git("rev-parse", f"{tag}^{{commit}}") != sha:
        raise ValueError(f"{tag} already points to a different commit")
    published = []
    for existing in tags:
        if existing.startswith("v"):
            try:
                published.append(Version(existing[1:]))
            except InvalidVersion:
                pass
    if published:
        latest = max(published)
        if Version(version) < latest or (
            Version(version) == latest and tag not in tags
        ):
            raise ValueError(f"Choose a version newer than {latest}")


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    version = requested_version(
        os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"], event
    )
    outputs = {"publish": "false"}
    if version:
        sha = git("rev-parse", "HEAD")
        if sha != os.environ["GITHUB_SHA"]:
            raise ValueError("Checkout does not match the triggering commit")
        validate_tags(version, sha)
        outputs.update(
            publish="true",
            version=version,
            tag=f"v{version}",
            sha=sha,
            prerelease=str(Version(version).is_prerelease).lower(),
        )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for name, value in outputs.items():
            output.write(f"{name}={value}\n")
    print(
        f"Requested release: v{version}" if version else "No release requested"
    )


if __name__ == "__main__":
    main()
