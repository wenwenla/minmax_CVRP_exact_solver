It is a python binding to solve minmax CVRP with small number of demands (n < 25).

**You may want to change the line 5 in CMakeLists.txt to use a different version of python/pybind11.**

Usage:

1. `mkdir build && cd build`
2. `cmake ..`
3. `make`


```python
import sys
sys.path.append('build/')
from dp_solver import mmcvrp_solver
import numpy as np

N = 21
m = 3
locs = np.random.uniform(0, 1, (N, 2)).astype(np.float32)
demands = np.random.randint(1, 9, (N, )).astype(np.int32)
demands[0] = 0
capacity = int(demands.sum() / m)

print(mmcvrp_solver(locs, demands, capacity, m))
```
