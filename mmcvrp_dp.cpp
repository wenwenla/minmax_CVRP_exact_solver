#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <cassert>

using namespace std;
namespace py = pybind11;


inline float sqr(float x) {
    return x * x;
}

inline float dist(const float* l, const float* r) {
    assert(sqr(l[0] - r[0]) + sqr(l[1] - r[1]) >= 0);
    return sqrt(sqr(l[0] - r[0]) + sqr(l[1] - r[1]));
}

std::vector<float> dp_tsp(const float* locs, const int* demands, int capacity, int n) {
    std::vector<int> weight_sum;
    std::vector<std::vector<float>> dp;
    std::queue<int> q;

    weight_sum.resize(1 << n);
    std::fill(weight_sum.begin(), weight_sum.end(), 0);
    q.push(1);
    while (!q.empty()) {
        int st = q.front();
        q.pop();
        for (int i = 0; i < n; ++i) {
            if (!(st >> i & 1)) {
                int nxt = st | ( 1 << i);
                if (weight_sum[nxt] == 0) {
                    q.push(nxt);
                    weight_sum[nxt] = weight_sum[st] + demands[i];
                }
            }
        }
    }
    dp.resize(n);
    for (int i = 0; i < n; ++i) {
        dp[i].resize(1 << n);
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < (1 << n); ++j) {
            dp[i][j] = 1e3;
        }
    }
    dp[0][1] = 0;
    for (int st = 0; st < (1 << n); ++st) {
        
        if (weight_sum[st] > capacity) {
            continue;
        }
        for (int i = 0; i < n; ++i) {
            if (!(st >> i & 1)) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                if (st >> j & 1) {
                    continue;
                }
                int nst = st | (1 << j);
                if (weight_sum[nst] > capacity) {
                    continue;
                }
                dp[j][nst] = min(dp[j][nst], dp[i][st] + dist(&locs[i * 2], &locs[j * 2]));
            }
        }
    }

    std::vector<float> result;
    result.resize(1 << n);
    for (int i = 0; i < (1 << n); ++i) {
        result[i] = 1e3;

        for (int j = 0; j < n; ++j) {
            result[i] = min(result[i], dp[j][i] + dist(&locs[0], &locs[j * 2]));
        }
    }
    return result;
}

float minmax_dp_cvrp(const std::vector<float>& cost, int n, int m) {
    std::vector<std::vector<float>> dp;
    dp.resize(m);
    for (int i = 0; i < m; ++i) {
        dp[i].resize((1 << n));
        std::fill(dp[i].begin(), dp[i].end(), 1e3);
    }
    for (int st = 0; st < (1 << n); ++st) {
        dp[0][st] = cost[st];
    }
    for (int i = 1; i < m; ++i) {
        for (int st = 1; st < (1 << n); st += 2) { // 0 must be selected
            int sub = st;
            while (sub) {
                int comp = sub ^ st;
                float this_cost = max(dp[i - 1][sub | 1], cost[comp | 1]);

                dp[i][st] = min(dp[i][st], this_cost);
                sub = (sub - 1) & st;
            }
        }
    }
    return dp[m - 1][(1 << n) - 1];
}


float mmcvrp_solver(py::array_t<float> locs, py::array_t<int> demands, int capacity, int m_agents) {
    auto locs_ref = locs.unchecked<2>();
    auto demands_ref = demands.unchecked<1>();
    int n_nodes = locs_ref.shape(0);

    auto locs_ptr = static_cast<const float*>(locs.data());
    auto demands_ptr = static_cast<const int*>(demands.data());

    auto cost = dp_tsp(locs_ptr, demands_ptr, capacity, n_nodes);
    float result = minmax_dp_cvrp(cost, n_nodes, m_agents);
    return result;
}

int main() {
    constexpr int N = 26;
    float locs[N * 2];
    int demands[N];
    int capacity;

    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> locs_dist(0.f, 1.f);
    std::uniform_int_distribution<int> demand_dist(1, 9);

    for (int i = 0; i < N; ++i) {
        demands[i] = demand_dist(rng);
        locs[i * 2] = locs_dist(rng);
        locs[i * 2 + 1] = locs_dist(rng);
    }

    for (int i = 0; i < N; ++i) {
        std::cout << i << " " << locs[i * 2] << " " << locs[i * 2 + 1] << std::endl;
    }

    demands[0] = 0;
    int total_demands = 0;
    for (int i = 0; i < N; ++i) {
        total_demands += demands[i];
    }
    capacity = ceil(total_demands / 3 * 1.2);

    auto cost = dp_tsp(locs, demands, capacity, N);
    std::cout << minmax_dp_cvrp(cost, n, 3) << std::endl;
    return 0;
}

PYBIND11_MODULE(dp_solver, m) {
    m.def("mmcvrp_solver", &mmcvrp_solver, "A function implemented in C++");
}