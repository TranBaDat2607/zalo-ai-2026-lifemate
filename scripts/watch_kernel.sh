#!/bin/bash
# watch_kernel.sh ACCOUNT NAME: poll the kernel's status every 5 minutes until it is no longer queued or running,
# riding out network errors (a DNS failure used to end `kg.py wait`), then print the final status.
cd "$(dirname "$0")/.." || exit 1
acc=$1; name=$2
while true; do
  s=$(timeout 90 python scripts/kg.py --account "$acc" status "$name" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1)
  case "$s" in
    *RUNNING*|*QUEUED*|"") ;;
    *) echo "$(date +%T) $acc $name: $s"; exit 0;;
  esac
  timeout 300 tail -f /dev/null
done
