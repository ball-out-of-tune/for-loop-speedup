import math
import time
import numpy as np

def hot_loop(data):
    ans = []
    for x in data:
        temp = math.cos(x) ** 2 + math.sin(x) ** 2 + math.sqrt(abs(x))
        ans.append(temp)

    return ans

def hot_loop_numpy(data):
    x = np.asarray(data)
    return np.sin(x) ** 2 + np.cos(x) ** 2 + np.sqrt(np.abs(x))

data1 = [i for i in range(1_000_000)]

if __name__ == '__main__':
    t0 = time.perf_counter()
    hot_loop(data1)
    gap = time.perf_counter() - t0
    print(f"纯 Python 耗时: {gap:.4f} 秒")

    t1 = time.perf_counter()
    hot_loop_numpy(data1)
    gap2 = time.perf_counter() - t1
    print(f"NumPy 耗时:    {gap2:.4f} 秒")

    print(f"NumPy 快了 {gap / gap2:.1f} 倍")
