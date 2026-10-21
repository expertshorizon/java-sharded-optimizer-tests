#!/bin/bash
set -uo pipefail

# Seal the reward channel before any agent-produced code can run. The pytest
# session below is root and can still write it; javac and every java process
# the session launches run as the unprivileged uid 10002 (through setpriv --no-new-privs in
# support.py), so neither the compiled CLI nor anything it forks off can read
# or overwrite reward.txt, before or after this script writes it. The directory
# is never opened up again.
mkdir -p /logs/verifier
chown root:root /logs/verifier
chmod 0700 /logs/verifier

chown root:root /tests
chmod 0750 /tests

python -m pytest --ctrf /logs/verifier/ctrf.json /tests -rA
rc=$?

if [ "$rc" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
