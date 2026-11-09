"""Does a named pipe's buffer size decide whether send_bytes waits for the
reader? A server sends 12 kB messages to a client that reads each one only
after 20 ms; the time send_bytes takes, with multiprocessing's default
8 kB pipe buffer and with 1 MB (set before the listener makes its pipes)."""
import sys
import threading
import time
from multiprocessing import connection


def client(address, n):
    c = connection.Client(address)
    for _ in range(n):
        time.sleep(0.02)                  # the reader is busy (its GIL elsewhere)
        c.recv_bytes()
    c.close()


def trial(bufsize, size=12 * 1024, n=20):
    connection.BUFSIZE = bufsize
    listener = connection.Listener()
    t = threading.Thread(target=client, args=(listener.address, n))
    t.start()
    conn = listener.accept()
    msg = b"x" * size
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        conn.send_bytes(msg)
        times.append((time.perf_counter() - t0) * 1e3)
        time.sleep(0.021)
    t.join()
    conn.close()
    listener.close()
    times.sort()
    return times[len(times) // 2], times[-1]


if __name__ == "__main__":
    assert sys.platform == "win32"
    for buf in (8192, 1 << 20, 8192, 1 << 20):
        med, worst = trial(buf)
        print(f"BUFSIZE {buf:>8}: send_bytes of 12 kB, median {med:.3f} ms, worst {worst:.3f} ms")
