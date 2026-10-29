# The server's pipes buffer a whole reply (2026-10-29)

**Question.** Stage 1's self-play spent 8% of its server's wall writing
replies (match s01: `reply` 331.9 of 4,091 s), about 0.17 ms a reply, with
the GPU idle meanwhile. The async-replies work (`feature/async-replies`,
`training/metrics/async_replies_20261029/`) found the cause:
- a reply larger than the named pipe's buffer, multiprocessing's 8 kB
  (`multiprocessing.connection.BUFSIZE`), completes only as its client
  reads it;
- a client reads only once its thread holds its worker process's GIL,
  which the other games of that worker hold during their steps;
- the replies average 8.5 kB, and 9 of the 20 quantiles of their sizes
  are larger.

So does a larger pipe buffer, set where the server makes its pipes, take
the writes off the serve thread's wall without a writer thread?

## The probe (`pipe_buf_probe.py`)

A server sends 12 kB messages to a client in the same process that reads
each one 20 ms late; the time `send_bytes` takes, 20 messages a trial,
alternating:

| buffer | median | worst |
|---|---|---|
| 8 kB (multiprocessing's) | 0.057, 0.067 ms | 20.4, 21.0 ms |
| 1 MB | 0.015, 0.016 ms | 0.023, 0.024 ms |

With 8 kB the write waits for the reader whenever the message does not fit;
with a buffer that holds the message it never does.

## The change (766022a6)

`tools/eval_inference_server.pipe_listener()` sets
`multiprocessing.connection.BUFSIZE` to 256 kB before making the Listener,
on Windows only; the module reads it only where it makes pipes. A reply
takes about 20 bytes a legal action, so 256 kB holds one of about 13,000
actions, and a connection has one reply outstanding at a time. Unix sockets
(the boxes) are unchanged; the bytes sent are the same.

Test: `tests/test_eval_inference_server.py::test_a_reply_larger_than_the_old_pipe_buffer_is_written_without_its_client`
(a 64 kB write returns while the client is not reading; with the old buffer
it would wait for the client's 10 s timeout).

## Self-play on the idle GPU

Stage 1 stopped after its match s02 (on its display-reset rule), and the
RX 6600 had nothing else on it from 15:04 to 15:33. `ab_idle.sh` played five
arms, 192 games each of `own2` against itself at temperature 1, 96 in
flight, eight a worker, seeds 872700, fp16 serving, in two rounds, the
second in reverse order. `ab_steady.py` reads each run's steady rate from
its server's timeline (10 s records, from 30 s to 40 s before the last).
Everything is in `ab_idle_results.txt`.

| arm | code | round 1 | round 2 | mean | against old |
|---|---|---|---|---|---|
| old | `feature/local-gpu-stage1` 041bd7ca | 489.0 | 481.2 | 485.1 | |
| bias | `feature/hip-bias-fold` 98e1ab1b | 521.1 | 524.0 | 522.6 | +7.7% |
| async | `feature/async-replies` 124ad8b4, `--inference-async-replies` | 507.1 | 501.6 | 504.4 | +4.0% |
| pipe | this branch, 766022a6 | 505.6 | 494.5 | 500.1 | +3.1% |
| bias and pipe | both merged (detached 6d0713a6) | 536.7 | 526.8 | 531.8 | **+9.6%** |

Steady decisions a second.
- **The two rounds of one arm** differ by up to 2.2% (pipe), so a difference
  of 2% between arms is at the noise.
- **The server's reply seconds** fall from 10.8-11.1 a run (old, bias) to
  4.9-5.3 (pipe, both). So about half the writes waited for their clients
  before. What is left is about 0.07 ms a reply, against the probe's 0.015;
  not explained.
- **Adopted:** the bias fold and the pipe buffer together. The pipe buffer
  takes most of what the writer thread takes (+3.1% against +4.0%, within
  the noise of each other). It needs no thread, does not split the batches,
  and changes nothing on the boxes.
- **The writer thread is not adopted:**
  - its benchmark found regimes where it loses (a forward costed per batch:
    -13%);
  - its timeline's GPU seconds read 17.6 of a 157 s run, a defect of its
    accounting under the thread, recorded on its branch.
