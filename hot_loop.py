import math
import time
import numpy as np
import numba
import statistics

def hot_loop(data):
    ans = []
    for x in data:
        temp = math.cos(x) ** 2 + math.sin(x) ** 2 + math.sqrt(abs(x))
        ans.append(temp)

    return ans

def hot_loop_numpy(data):
    x = np.asarray(data)
    return np.sin(x) ** 2 + np.cos(x) ** 2 + np.sqrt(np.abs(x))

@numba.njit
def hot_loop_numba(data_np):
    n = len(data_np)
    ans = np.empty(n)
    for i in range(n):
        tmp = data_np[i]
        ans[i] = math.sin(tmp) ** 2 + math.cos(tmp) ** 2 + math.sqrt(abs(tmp))
    return ans


@numba.njit(parallel=True)
def hot_loop_numba_parallel(data_np):
    n = len(data_np)
    ans = np.empty(n)
    for i in numba.prange(n):
        tmp = data_np[i]
        ans[i] = math.sin(tmp) ** 2 + math.cos(tmp) ** 2 + math.sqrt(abs(tmp))
    return ans

@numba.njit
def hot_loop_numba_heavy(data_np):
    n = len(data_np)
    ans = np.empty(n)
    for i in range(n):
        x = data_np[i]
        for _ in range(100):
            x = math.sin(x) + math.cos(x) * math.sqrt(abs(x) + 1)
        ans[i] = x
    return ans


@numba.njit(parallel=True)
def hot_loop_numba_parallel_heavy(data_np):
    n = len(data_np)
    ans = np.empty(n)
    for i in numba.prange(n):
        x = data_np[i]
        for _ in range(100):
            x = math.sin(x) + math.cos(x) * math.sqrt(abs(x) + 1)
        ans[i] = x
    return ans


data1 = [i for i in range(1_000_000)]

if __name__ == '__main__':
    warmup_steps = 3
    warmup_time_python_loop = []
    for _ in range(warmup_steps):
        t0 = time.perf_counter()
        hot_loop(data1)
        gap = time.perf_counter() - t0
        warmup_time_python_loop.append(gap)
        
    print(f"warmup_mean_time : {statistics.mean(warmup_time_python_loop)}")

    time_gap_python_loop = []
    for _ in range(10):
        t_python_loop1 = time.perf_counter()
        hot_loop(data1)
        gap_python_loop = time.perf_counter() - t_python_loop1
        time_gap_python_loop.append(gap_python_loop)
    print(f"statistics.mean : {statistics.mean(time_gap_python_loop)}")

    warmup_time_numpy = []
    for _ in range(warmup_steps):
        t0 = time.perf_counter()
        hot_loop_numpy(data1)
        gap = time.perf_counter() - t0
        warmup_time_numpy.append(gap)
    print(f"numpy warmup_mean_time : {statistics.mean(warmup_time_numpy)}")

    time_gap_numpy = []
    for _ in range(10):
        t0 = time.perf_counter()
        hot_loop_numpy(data1)
        gap = time.perf_counter() - t0
        time_gap_numpy.append(gap)
    print(f"numpy statistics.mean : {statistics.mean(time_gap_numpy)}")

    t2 = time.perf_counter()
    hot_loop_numba(np.asarray(data1))
    compilation_time = time.perf_counter() - t2
    print(f"numba compilation_time : {compilation_time}")

    warmup_time_numba = []
    for _ in range(warmup_steps):
        t0 = time.perf_counter()
        hot_loop_numba(np.asarray(data1))
        gap = time.perf_counter() - t0
        warmup_time_numba.append(gap)
    print(f"numba warmup_mean_time : {statistics.mean(warmup_time_numba)}")

    time_gap_numba = []
    for _ in range(10):
        t0 = time.perf_counter()
        hot_loop_numba(np.asarray(data1))
        gap = time.perf_counter() - t0
        time_gap_numba.append(gap)
    print(f"numba statistics.mean : {statistics.mean(time_gap_numba)}")

    t3 = time.perf_counter()
    hot_loop_numba_parallel(np.asarray(data1))
    compilation_time_parallel = time.perf_counter() - t3
    print(f"numba parallel compilation_time : {compilation_time_parallel}")

    warmup_time_numba_parallel = []
    for _ in range(warmup_steps):
        t0 = time.perf_counter()
        hot_loop_numba_parallel(np.asarray(data1))
        gap = time.perf_counter() - t0
        warmup_time_numba_parallel.append(gap)
    print(f"numba parallel warmup_mean_time : {statistics.mean(warmup_time_numba_parallel)}")

    time_gap_numba_parallel = []
    for _ in range(10):
        t0 = time.perf_counter()
        hot_loop_numba_parallel(np.asarray(data1))
        gap = time.perf_counter() - t0
        time_gap_numba_parallel.append(gap)
    print(f"numba parallel statistics.mean : {statistics.mean(time_gap_numba_parallel)}")

    # --- heavy 版本：计算量 100 倍，内存读写不变 ---
    t4 = time.perf_counter()
    hot_loop_numba_heavy(np.asarray(data1))
    compilation_time_heavy = time.perf_counter() - t4
    print(f"numba heavy compilation_time : {compilation_time_heavy}")

    warmup_time_numba_heavy = []
    for _ in range(warmup_steps):
        t0 = time.perf_counter()
        hot_loop_numba_heavy(np.asarray(data1))
        gap = time.perf_counter() - t0
        warmup_time_numba_heavy.append(gap)
    print(f"numba heavy warmup_mean_time : {statistics.mean(warmup_time_numba_heavy)}")

    time_gap_numba_heavy = []
    for _ in range(10):
        t0 = time.perf_counter()
        hot_loop_numba_heavy(np.asarray(data1))
        gap = time.perf_counter() - t0
        time_gap_numba_heavy.append(gap)
    print(f"numba heavy statistics.mean : {statistics.mean(time_gap_numba_heavy)}")

    # --- heavy + parallel ---
    t5 = time.perf_counter()
    hot_loop_numba_parallel_heavy(np.asarray(data1))
    compilation_time_parallel_heavy = time.perf_counter() - t5
    print(f"numba parallel heavy compilation_time : {compilation_time_parallel_heavy}")

    warmup_time_numba_parallel_heavy = []
    for _ in range(warmup_steps):
        t0 = time.perf_counter()
        hot_loop_numba_parallel_heavy(np.asarray(data1))
        gap = time.perf_counter() - t0
        warmup_time_numba_parallel_heavy.append(gap)
    print(f"numba parallel heavy warmup_mean_time : {statistics.mean(warmup_time_numba_parallel_heavy)}")

    time_gap_numba_parallel_heavy = []
    for _ in range(10):
        t0 = time.perf_counter()
        hot_loop_numba_parallel_heavy(np.asarray(data1))
        gap = time.perf_counter() - t0
        time_gap_numba_parallel_heavy.append(gap)
    print(f"numba parallel heavy statistics.mean : {statistics.mean(time_gap_numba_parallel_heavy)}")