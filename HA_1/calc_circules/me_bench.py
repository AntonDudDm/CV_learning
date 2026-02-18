import time

def bench(fn, n=20, warmup=3):
    # прогрев
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / n
    print(f"avg: {avg*1000:.3f} ms, min: {min(times)*1000:.3f} ms, max: {max(times)*1000:.3f} ms")