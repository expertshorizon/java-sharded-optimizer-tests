#!/bin/bash
set -e

SOLUTION_DIR=/solution
if [ ! -d "$SOLUTION_DIR/src" ]; then
    SOLUTION_DIR=/oracle
fi

rm -rf /app/src
cp -r "$SOLUTION_DIR/src" /app/src

mkdir -p /app/out
javac -d /app/out $(find /app/src/main/java -name '*.java')
