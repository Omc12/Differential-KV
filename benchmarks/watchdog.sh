#!/usr/bin/env bash
# Every 30 minutes: is the campaign alive, and is the data still clean?
#
# WHY: this campaign has produced seven silent contaminations, and every one was
# caught by someone happening to look. A run that keeps going while writing
# wrong numbers costs more than a run that stops. This checks the two things
# that actually go wrong -- the campaign dying quietly, and result files
# failing the audit -- and prints a line ONLY when there is something to act on.
#
# Silence means healthy. Each printed line is a notification, so it must be
# worth interrupting for.

set -u
cd "$(dirname "$0")/.." || exit 1
PY="C:/Users/USER/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

LOG="paper/results/campaign_resume.log"
INTERVAL="${WATCHDOG_INTERVAL:-1800}"
last_rows=-1
stalls=0

while true; do
  sleep "$INTERVAL"

  # 1. FAILING RESULT FILES -- the thing that silently poisons a table.
  fails=$("$PY" benchmarks/validate_results.py 2>/dev/null | grep -c '^\[ FAIL ')
  if [ "${fails:-0}" -gt 0 ]; then
    echo "WATCHDOG: $fails result file(s) FAIL the audit — run benchmarks/validate_results.py"
  fi

  # 2. IS IT STILL MOVING? Total recorded rows across every store. Two
  #    consecutive checks with no new rows and no python running means the
  #    chain died; the campaign log's own tail will not say so.
  rows=$(cat paper/results/*/*.jsonl 2>/dev/null | grep -c . || echo 0)
  alive=$(tasklist //FI "IMAGENAME eq python.exe" //FO CSV //NH 2>/dev/null | grep -c python || echo 0)
  if [ "$rows" = "$last_rows" ] && [ "${alive:-0}" -eq 0 ]; then
    stalls=$((stalls + 1))
    if [ "$stalls" -ge 2 ]; then
      echo "WATCHDOG: campaign appears STOPPED — $rows rows, no python process, no growth in $((INTERVAL*2/60)) min"
      stalls=0
    fi
  else
    stalls=0
  fi
  last_rows=$rows

  # 3. CONCURRENT WRITERS -- two processes on one GPU make every memory and
  #    latency number contended, and nothing in the data marks it.
  locks=$(ls paper/results/*/*.lock 2>/dev/null | wc -l)
  if [ "${locks:-0}" -gt 1 ]; then
    echo "WATCHDOG: $locks result files locked at once — concurrent writers contend for the GPU"
  fi

  # 4. ORPHANED CHAINS. A killed chain's bash can survive, blocked forever on a
  #    log marker that is no longer written -- harmless until the day that
  #    string appears and it launches an arm alongside the live campaign. Two
  #    were found this way. Counts DISTINCT chain scripts; a parent/child bash
  #    pair of the same script is normal and is not double-counted.
  # EXPECTED_CHAINS is the number of chains deliberately queued and waiting.
  # Staged chains are normal here: each waits on the previous one's completion
  # marker, so several are alive at once by design and only ONE ever holds the
  # GPU. A threshold that ignores that fires on a healthy campaign, which is
  # worse than not checking -- an alarm nobody believes is an alarm nobody
  # reads. Set EXPECTED_CHAINS when the queue changes.
  distinct=$(ps -W 2>/dev/null | grep -o 'chain_[a-z0-9_]*\.sh' | sort -u | wc -l)
  if [ "${distinct:-0}" -gt "${EXPECTED_CHAINS:-3}" ]; then
    echo "WATCHDOG: $distinct distinct chain scripts alive, expected ${EXPECTED_CHAINS:-3} — an orphan from a killed chain may fire unexpectedly"
  fi

  # 4b. THE CHECK THAT ACTUALLY MATTERS: more than one GPU worker. Chains are
  #     harmless while they wait; two python processes on one card are not.
  workers=$(tasklist //FI "IMAGENAME eq python.exe" //FO CSV //NH 2>/dev/null | grep -c python || echo 0)
  if [ "${workers:-0}" -gt 1 ]; then
    echo "WATCHDOG: $workers python workers running at once — they share one GPU and every memory/latency number either records is contended"
  fi

  # 5. A finished campaign is worth one line.
  if grep -aq "ALL CAMPAIGNS DONE" "$LOG" 2>/dev/null; then
    echo "WATCHDOG: ALL CAMPAIGNS DONE — $rows rows recorded, $fails failing files"
    exit 0
  fi
done
