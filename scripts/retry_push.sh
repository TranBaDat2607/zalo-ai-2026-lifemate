#!/bin/bash
# retry_push.sh ACCOUNT OWNER BASE "PUSH OPTIONS" PLAN DEADLINE_HOURS
# Push OWNER/lifemate-BASE as a jobs run of PLAN; while it sits QUEUED for 2 minutes, delete it and push BASE-r2,
# BASE-r3, ... (up to 30 tries). Prints "STARTED <slug>" once one is RUNNING (or already finished).
# Example: bash scripts/retry_push.sh main OWNER view-bench3 "--hours 1.0 --dataset OWNER/lifemate-asr-avg5-full" \
#          scripts/plans/view_bench.json 0.85
cd "$(dirname "$0")/.." || exit 1
acc=$1; owner=$2; base=$3; opts=$4; plan=$5; deadline=$6
st() { timeout 60 python scripts/kg.py --account "$acc" status "$1" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1; }
for n in $(seq 1 30); do
  slug=$base
  [ "$n" -gt 1 ] && slug=$base-r$n
  # shellcheck disable=SC2086  (the push options are several words on purpose)
  timeout 300 python scripts/kg.py --account "$acc" push $opts "$slug" jobs -- --plan "$(tr -d '\n' < "$plan")" \
    --deadline-hours "$deadline" 2>&1 | grep -v -i warn | grep -E "pushed|rror" | tail -1
  for _ in 1 2 3 4; do
    timeout 30 tail -f /dev/null
    s=$(st "$slug")
    case "$s" in *RUNNING*|*COMPLETE*|*ERROR*) echo "$(date +%T) $owner/lifemate-$slug: $s"; echo "STARTED $slug"; exit 0;; esac
  done
  echo "$(date +%T) $slug still ${s:-UNKNOWN}, deleting"
  timeout 120 python scripts/kg.py --account "$acc" cli kernels delete -y "$owner/lifemate-$slug" 2>&1 | grep -v -i warn | tail -1
done
echo "GAVE UP after 30 tries"
exit 1
