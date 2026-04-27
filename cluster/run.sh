#!/usr/bin/env bash
# Wrapper around cluster/job.sbatch — submit/status/tail/stop without
# memorizing SLURM commands.
#
# Usage:
#   bash cluster/run.sh           # submit a fresh training job
#   bash cluster/run.sh status    # squeue + GPU util + last log lines
#   bash cluster/run.sh tail      # follow the log live
#   bash cluster/run.sh stop      # scancel the latest submitted job

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/training/logs"
JOBID_FILE="${LOG_DIR}/supervised.jobid"
mkdir -p "${LOG_DIR}"

current_jobid() {
    [ -f "${JOBID_FILE}" ] && cat "${JOBID_FILE}" || echo ""
}

current_log() {
    local jid; jid="$(current_jobid)"
    [ -n "${jid}" ] && echo "${LOG_DIR}/slurm-${jid}.log" || echo ""
}

case "${1:-start}" in
    status)
        jid="$(current_jobid)"
        echo "--- queue (this user) ---"
        squeue --me 2>/dev/null || true
        if [ -n "${jid}" ]; then
            echo "--- our job ${jid} ---"
            squeue -j "${jid}" --noheader -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" 2>/dev/null \
                || echo "  (job ${jid} no longer in the queue)"
            log="$(current_log)"
            if [ -f "${log}" ]; then
                echo "--- last 15 log lines ($(basename "${log}")) ---"
                tail -15 "${log}"
            fi
        else
            echo "no submitted job recorded (no ${JOBID_FILE})"
        fi
        ;;
    tail)
        log="$(current_log)"
        if [ -z "${log}" ] || [ ! -f "${log}" ]; then
            echo "no log file yet (job hasn't started, or never submitted)" >&2
            exit 1
        fi
        tail -F "${log}"
        ;;
    stop)
        jid="$(current_jobid)"
        if [ -z "${jid}" ]; then
            echo "no recorded job id"
            exit 1
        fi
        echo "scancel ${jid}"
        scancel "${jid}"
        ;;
    start)
        jid="$(current_jobid)"
        if [ -n "${jid}" ] \
           && squeue -j "${jid}" --noheader 2>/dev/null | grep -q .; then
            echo "training already submitted (jobid ${jid}); refusing to double-submit" >&2
            echo "use 'stop' to cancel it first, or 'status' / 'tail' to monitor" >&2
            exit 1
        fi
        out=$(sbatch cluster/job.sbatch)
        # `sbatch` prints "Submitted batch job 12345"; capture the id.
        new_jid="$(echo "${out}" | awk '{print $NF}')"
        echo "${new_jid}" > "${JOBID_FILE}"
        echo "submitted: ${out}"
        echo "log will be: ${LOG_DIR}/slurm-${new_jid}.log"
        echo "monitor:    bash cluster/run.sh status"
        echo "tail:       bash cluster/run.sh tail"
        echo "cancel:     bash cluster/run.sh stop"
        ;;
    *)
        echo "usage: $0 [start|status|tail|stop]" >&2
        exit 2 ;;
esac
