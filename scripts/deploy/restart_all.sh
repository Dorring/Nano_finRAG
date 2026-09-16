#!/usr/bin/env bash
# Restart all Phase 7 services: stop, then start.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${__DIR}/stop_all.sh"
"${__DIR}/start_all.sh"
