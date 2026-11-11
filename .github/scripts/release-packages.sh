#!/usr/bin/env bash
set -euo pipefail

# Maps a release tag such as `v0.4.9` to the workspace's publishable crates and
# checks that every crate's manifest version matches the tag. Prints
# `packages=...` for `$GITHUB_OUTPUT`.

TAG="${1:?usage: release-packages.sh <tag>}"
VERSION="${TAG#v}"

if [[ "$TAG" != v* || -z "$VERSION" ]]; then
    echo "error: tag '${TAG}' is not of the form v<version>" >&2
    exit 1
fi

metadata="$(cargo metadata --format-version 1 --no-deps --locked)"
packages="$(jq -r '.packages[] | select(.publish != []) | .name' <<<"$metadata")"
if [[ -z "$packages" ]]; then
    echo "error: no publishable packages in the workspace" >&2
    exit 1
fi

for package in $packages; do
    actual="$(jq -r --arg name "$package" '.packages[] | select(.name == $name) | .version' <<<"$metadata")"
    if [[ "$actual" != "$VERSION" ]]; then
        echo "error: ${package} is ${actual}, but tag ${TAG} releases ${VERSION}" >&2
        exit 1
    fi
done

echo "packages=$(paste -sd " " <<<"$packages")"
