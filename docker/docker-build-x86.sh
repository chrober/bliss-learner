#!/bin/bash
## #!/usr/bin/env bash
set -eux

uname -a
DESTDIR=/src/releases
mkdir -p $DESTDIR

function build {
    echo Building for $1 to $3...

    if [[ ! -f /build/$1/release/bliss-learner ]]; then
        cargo build --release --target $1
    fi

    $2 /build/$1/release/bliss-learner && cp /build/$1/release/bliss-learner $DESTDIR/$3
}

build x86_64-unknown-linux-musl strip bliss-learner
