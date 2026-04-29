#!/usr/bin/env bash
# Wrapper around cluster/job.sbatch / cluster/job_selfplay.sbatch —
# submit/status/tail/stop without memorizing SLURM commands.
#
# Usage:
#   bash cluster/run.sh                          # submit a supervised job
#   bash cluster/run.sh start                    # idem
#   bash cluster/run.sh start supervised         # idem
#   bash cluster/run.sh start selfplay           # submit a self-play job
#   bash cluster/run.sh continue [<mode>]        # cancel running ${mode}
#                                                # job (if any) and submit
#                                                # a fresh one that auto-
#                                                # resumes from the latest
#                                                # checkpoint. Used after
#                                                # walltime; replaces the
#                                                # old auto-resubmit chain.
#   bash cluster/run.sh status                   # show both modes if present
#   bash cluster/run.sh status supervised        # only the supervised job
#   bash cluster/run.sh tail [supervised|selfplay]
#   bash cluster/run.sh stop [supervised|selfplay]   # default supervised
#   bash cluster/run.sh budget                   # squeue + today's sacct +
#                                                # 7-day summary + sshare
#
# Backwards compat: omitting the second arg keeps the old supervised-
# only behavior, except for `status` which now shows both modes.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/training/logs"
mkdir -p "${LOG_DIR}"

# Per-mode config: which jobid file, which sbatch script, which slurm
# log prefix (matches the corresponding sbatch's `--output` line).
mode_jobid_file() {
    case "$1" in
        supervised) echo "${LOG_DIR}/supervised.jobid" ;;
        selfplay)   echo "${LOG_DIR}/selfplay.jobid"   ;;
        *) echo "unknown mode: $1" >&2; exit 2 ;;
    esac
}
mode_sbatch_file() {
    case "$1" in
        supervised) echo "cluster/job.sbatch"          ;;
        selfplay)   echo "cluster/job_selfplay.sbatch" ;;
        *) echo "unknown mode: $1" >&2; exit 2 ;;
    esac
}
mode_log_prefix() {
    case "$1" in
        supervised) echo "slurm-"          ;;
        selfplay)   echo "selfplay-slurm-" ;;
        *) echo "unknown mode: $1" >&2; exit 2 ;;
    esac
}

current_jobid() {
    # $1: mode
    local f; f="$(mode_jobid_file "$1")"
    [ -f "${f}" ] && cat "${f}" || echo ""
}

current_log() {
    # $1: mode
    local jid; jid="$(current_jobid "$1")"
    [ -z "${jid}" ] && { echo ""; return; }
    echo "${LOG_DIR}/$(mode_log_prefix "$1")${jid}.log"
}

show_status_for_mode() {
    # $1: mode
    local mode="$1"
    local jid; jid="$(current_jobid "${mode}")"
    if [ -z "${jid}" ]; then
        echo "no submitted ${mode} job recorded (no $(mode_jobid_file "${mode}"))"
        return
    fi
    echo "--- our ${mode} job ${jid} ---"
    squeue -j "${jid}" --noheader -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" 2>/dev/null \
        || echo "  (job ${jid} no longer in the queue)"
    local log; log="$(current_log "${mode}")"
    if [ -f "${log}" ]; then
        echo "--- last 15 log lines ($(basename "${log}")) ---"
        tail -15 "${log}"
    fi
}

CMD="${1:-start}"
MODE="${2:-supervised}"

case "${CMD}" in
    status)
        echo "--- queue (this user) ---"
        squeue --me 2>/dev/null || true
        # When no mode is given, show both — operators commonly run
        # supervised + self-play jobs in the same project tree and
        # want a single glance.
        if [ -n "${2:-}" ]; then
            show_status_for_mode "${MODE}"
        else
            show_status_for_mode "supervised"
            show_status_for_mode "selfplay"
        fi
        ;;
    tail)
        log="$(current_log "${MODE}")"
        if [ -z "${log}" ] || [ ! -f "${log}" ]; then
            echo "no ${MODE} log file yet (job hasn't started, or never submitted)" >&2
            exit 1
        fi
        tail -F "${log}"
        ;;
    stop)
        jid="$(current_jobid "${MODE}")"
        if [ -z "${jid}" ]; then
            echo "no recorded ${MODE} job id"
            exit 1
        fi
        echo "scancel ${jid}"
        scancel "${jid}"
        ;;
    start)
        jid_file="$(mode_jobid_file "${MODE}")"
        sbatch_file="$(mode_sbatch_file "${MODE}")"
        jid="$(current_jobid "${MODE}")"
        if [ -n "${jid}" ] \
           && squeue -j "${jid}" --noheader 2>/dev/null | grep -q .; then
            echo "${MODE} training already submitted (jobid ${jid}); " \
                 "refusing to double-submit" >&2
            echo "use 'stop ${MODE}' to cancel it first, " \
                 "or 'status' / 'tail ${MODE}' to monitor" >&2
            exit 1
        fi
        out=$(sbatch "${sbatch_file}")
        # `sbatch` prints "Submitted batch job 12345"; capture the id.
        new_jid="$(echo "${out}" | awk '{print $NF}')"
        echo "${new_jid}" > "${jid_file}"
        echo "submitted: ${out}"
        log_prefix="$(mode_log_prefix "${MODE}")"
        echo "log will be: ${LOG_DIR}/${log_prefix}${new_jid}.log"
        echo "monitor:    bash cluster/run.sh status"
        echo "tail:       bash cluster/run.sh tail ${MODE}"
        echo "cancel:     bash cluster/run.sh stop ${MODE}"
        ;;
    continue)
        # Cancel any running ${MODE} job, then sbatch a fresh one
        # (which auto-resumes from the latest checkpoint via the
        # sbatch script's RESUME / latest-ckpt logic). Used after
        # a walltime-cancelled job to start the next link manually
        # -- replaces the old auto-resubmit chain. Polite on shared
        # clusters: each link is operator-initiated.
        jid_file="$(mode_jobid_file "${MODE}")"
        sbatch_file="$(mode_sbatch_file "${MODE}")"
        jid="$(current_jobid "${MODE}")"
        if [ -n "${jid}" ] \
           && squeue -j "${jid}" --noheader 2>/dev/null | grep -q .; then
            echo "scancel ${jid} (cancelling running ${MODE} job before continue)"
            scancel "${jid}"
            # Brief settle so the new sbatch sees an open slot.
            sleep 2
        fi
        out=$(sbatch "${sbatch_file}")
        new_jid="$(echo "${out}" | awk '{print $NF}')"
        echo "${new_jid}" > "${jid_file}"
        echo "submitted: ${out}"
        log_prefix="$(mode_log_prefix "${MODE}")"
        echo "log will be: ${LOG_DIR}/${log_prefix}${new_jid}.log"
        echo "monitor:    bash cluster/run.sh status"
        echo "tail:       bash cluster/run.sh tail ${MODE}"
        echo "cancel:     bash cluster/run.sh stop ${MODE}"
        ;;
    budget)
        # Snapshot of cluster usage / budget. Best-effort: we run
        # the standard SLURM accounting commands and let the
        # operator interpret. Some clusters don't expose all of
        # these (sshare in particular needs accounting enabled).
        echo "--- queue (this user) ---"
        squeue --me 2>/dev/null || echo "(squeue unavailable)"
        echo
        echo "--- today's jobs (sacct since 00:00) ---"
        sacct -u "${USER}" -X \
            --starttime="$(date -I)" \
            --format=JobID,JobName%18,Partition%14,State,Start,Elapsed,ExitCode \
            2>/dev/null \
            || echo "(sacct unavailable)"
        echo
        echo "--- last 7 days summary ---"
        sacct -u "${USER}" -X \
            --starttime=now-7days \
            --format=JobID,State,Elapsed,Partition \
            -P 2>/dev/null \
            | awk -F'|' 'NR>1 && $2 != "" {
                state=$2;
                # Elapsed is HH:MM:SS or D-HH:MM:SS.
                t=$3; n=split(t, p, /[-:]/);
                if (n==4) sec=((p[1]*24)+p[2])*3600+p[3]*60+p[4];
                else      sec=p[1]*3600+p[2]*60+p[3];
                total+=sec; by_state[state]+=sec; count[state]++;
              }
              END {
                for (s in by_state) {
                  printf "  %-12s %4d jobs  %5.1f h\n",
                    s, count[s], by_state[s]/3600.0;
                }
                printf "  %-12s        %5.1f h total\n", "ALL",
                       total/3600.0;
              }' \
            || echo "(sacct/awk unavailable)"
        echo
        echo "--- fair-share (sshare) ---"
        sshare -U 2>/dev/null \
            || echo "(sshare unavailable -- accounting may not be configured)"
        ;;
    *)
        echo "usage: $0 [start|continue|status|tail|stop|budget] [supervised|selfplay]" >&2
        exit 2 ;;
esac
