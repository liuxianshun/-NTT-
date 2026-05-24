#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(HAVE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#else
static int omp_get_max_threads() { return 1; }
static void omp_set_num_threads(int) {}
static void omp_set_nested(int) {}
#endif

#if !defined(_WIN32)
#include <pthread.h>
#define HAVE_PTHREAD_IMPL 1
#endif

namespace bench {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 MOD_A = 998244353u;
constexpr u32 ROOT_A = 3u;
constexpr u32 MOD_B = 1004535809u;
constexpr u32 ROOT_B = 3u;
constexpr u32 MOD_C = 469762049u;
constexpr u32 ROOT_C = 3u;
constexpr u32 TARGET_MOD = 1000000007u;

template <class T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <class T, class U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return true;
}

template <class T, class U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return false;
}

struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point start = clock::now();

    double elapsed_ms() const {
        const auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

u32 add_mod(u32 a, u32 b, u32 mod) {
    const u32 s = a + b;
    return (s >= mod || s < a) ? s - mod : s;
}

u32 sub_mod(u32 a, u32 b, u32 mod) {
    return (a >= b) ? (a - b) : (a + mod - b);
}

u32 mul_mod(u32 a, u32 b, u32 mod) {
    return static_cast<u32>((static_cast<u64>(a) * b) % mod);
}

u32 pow_mod(u32 a, u64 e, u32 mod) {
    u64 result = 1;
    u64 base = a;
    while (e > 0) {
        if (e & 1u) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        e >>= 1u;
    }
    return static_cast<u32>(result);
}

u64 inverse_mod_u64(u64 a, u64 mod) {
    long long t = 0;
    long long new_t = 1;
    long long r = static_cast<long long>(mod);
    long long new_r = static_cast<long long>(a % mod);
    while (new_r != 0) {
        const long long q = r / new_r;
        const long long next_t = t - q * new_t;
        t = new_t;
        new_t = next_t;
        const long long next_r = r - q * new_r;
        r = new_r;
        new_r = next_r;
    }
    if (r > 1) {
        throw std::runtime_error("inverse does not exist");
    }
    if (t < 0) {
        t += static_cast<long long>(mod);
    }
    return static_cast<u64>(t);
}

std::size_t ceil_pow2(std::size_t n) {
    std::size_t x = 1;
    while (x < n) {
        x <<= 1u;
    }
    return x;
}

std::vector<u32> random_poly(std::size_t n, u32 max_value, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<u32> dist(0, max_value);
    std::vector<u32> a(n);
    for (auto& x : a) {
        x = dist(rng);
    }
    return a;
}

u64 checksum(const std::vector<u32>& v) {
    u64 h = 1469598103934665603ull;
    for (u32 x : v) {
        h ^= x;
        h *= 1099511628211ull;
    }
    return h;
}

template <u32 MOD>
std::vector<u32> direct_scalar(const std::vector<u32>& a, const std::vector<u32>& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<u32> c(n + m - 1, 0);
    for (std::size_t k = 0; k < c.size(); ++k) {
        const std::size_t begin = (k >= m - 1) ? (k - (m - 1)) : 0;
        const std::size_t end = std::min(k, n - 1) + 1;
        u64 acc = 0;
        for (std::size_t i = begin; i < end; ++i) {
            acc += static_cast<u64>(a[i]) * b[k - i];
            if (acc > (1ull << 62)) {
                acc %= MOD;
            }
        }
        c[k] = static_cast<u32>(acc % MOD);
    }
    return c;
}

template <u32 MOD>
std::vector<u32> direct_omp(const std::vector<u32>& a,
                            const std::vector<u32>& b,
                            int threads,
                            bool dynamic_schedule) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<u32> c(n + m - 1, 0);
    const long long total = static_cast<long long>(c.size());

    auto work = [&](long long k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        const std::size_t begin = (kk >= m - 1) ? (kk - (m - 1)) : 0;
        const std::size_t end = std::min(kk, n - 1) + 1;
        u64 acc = 0;
        for (std::size_t i = begin; i < end; ++i) {
            acc += static_cast<u64>(a[i]) * b[kk - i];
            if (acc > (1ull << 62)) {
                acc %= MOD;
            }
        }
        c[kk] = static_cast<u32>(acc % MOD);
    };

    if (dynamic_schedule) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
        for (long long k = 0; k < total; ++k) {
            work(k);
        }
    } else {
#pragma omp parallel for num_threads(threads) schedule(static)
        for (long long k = 0; k < total; ++k) {
            work(k);
        }
    }
    return c;
}

#if defined(HAVE_PTHREAD_IMPL)
template <u32 MOD>
struct PThreadConvTask {
    const std::vector<u32>* a = nullptr;
    const std::vector<u32>* b = nullptr;
    std::vector<u32>* c = nullptr;
    std::size_t begin_k = 0;
    std::size_t end_k = 0;
};

template <u32 MOD>
void* pthread_conv_worker(void* raw) {
    auto* task = static_cast<PThreadConvTask<MOD>*>(raw);
    const auto& a = *task->a;
    const auto& b = *task->b;
    auto& c = *task->c;
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    for (std::size_t k = task->begin_k; k < task->end_k; ++k) {
        const std::size_t begin = (k >= m - 1) ? (k - (m - 1)) : 0;
        const std::size_t end = std::min(k, n - 1) + 1;
        u64 acc = 0;
        for (std::size_t i = begin; i < end; ++i) {
            acc += static_cast<u64>(a[i]) * b[k - i];
            if (acc > (1ull << 62)) {
                acc %= MOD;
            }
        }
        c[k] = static_cast<u32>(acc % MOD);
    }
    return nullptr;
}

template <u32 MOD>
std::vector<u32> direct_pthread(const std::vector<u32>& a, const std::vector<u32>& b, int threads) {
    std::vector<u32> c(a.size() + b.size() - 1, 0);
    std::vector<pthread_t> handles(static_cast<std::size_t>(threads));
    std::vector<PThreadConvTask<MOD>> tasks(static_cast<std::size_t>(threads));
    const std::size_t chunk = (c.size() + static_cast<std::size_t>(threads) - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        tasks[static_cast<std::size_t>(t)] = {&a, &b, &c, static_cast<std::size_t>(t) * chunk,
                                              std::min(c.size(), static_cast<std::size_t>(t + 1) * chunk)};
        pthread_create(&handles[static_cast<std::size_t>(t)], nullptr, pthread_conv_worker<MOD>,
                       &tasks[static_cast<std::size_t>(t)]);
    }
    for (auto& handle : handles) {
        pthread_join(handle, nullptr);
    }
    return c;
}
#endif

bool is_aligned_32(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 31u) == 0;
}

template <u32 MOD>
std::vector<u32> direct_avx2(const std::vector<u32>& a,
                             const std::vector<u32>& b,
                             bool prefer_aligned_loads) {
#if defined(__AVX2__)
    using AlignedU64 = std::vector<u64, AlignedAllocator<u64, 32>>;
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    AlignedU64 aa(n);
    AlignedU64 br(m);
    for (std::size_t i = 0; i < n; ++i) {
        aa[i] = a[i];
    }
    for (std::size_t i = 0; i < m; ++i) {
        br[i] = b[m - 1 - i];
    }

    std::vector<u32> c(n + m - 1, 0);
    for (std::size_t k = 0; k < c.size(); ++k) {
        const std::size_t begin = (k >= m - 1) ? (k - (m - 1)) : 0;
        const std::size_t end = std::min(k, n - 1) + 1;
        const std::size_t br_base = (k >= m - 1) ? 0 : (m - 1 - k);
        std::size_t i = begin;
        __m256i vacc = _mm256_setzero_si256();
        for (; i + 4 <= end; i += 4) {
            const auto* pa = reinterpret_cast<const __m256i*>(&aa[i]);
            const auto* pb = reinterpret_cast<const __m256i*>(&br[br_base + (i - begin)]);
            const bool aligned = prefer_aligned_loads && is_aligned_32(pa) && is_aligned_32(pb);
            const __m256i va = aligned ? _mm256_load_si256(pa) : _mm256_loadu_si256(pa);
            const __m256i vb = aligned ? _mm256_load_si256(pb) : _mm256_loadu_si256(pb);
            vacc = _mm256_add_epi64(vacc, _mm256_mul_epu32(va, vb));
        }
        alignas(32) u64 lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), vacc);
        u64 acc = (lanes[0] % MOD + lanes[1] % MOD + lanes[2] % MOD + lanes[3] % MOD) % MOD;
        for (; i < end; ++i) {
            acc += static_cast<u64>(a[i]) * b[k - i];
            if (acc > (1ull << 62)) {
                acc %= MOD;
            }
        }
        c[k] = static_cast<u32>(acc % MOD);
    }
    return c;
#else
    (void)prefer_aligned_loads;
    return direct_scalar<MOD>(a, b);
#endif
}

template <u32 MOD, u32 ROOT>
void ntt(std::vector<u32>& a, bool invert, int threads, bool dynamic_schedule) {
    const std::size_t n = a.size();
    if ((MOD - 1) % n != 0) {
        throw std::runtime_error("NTT size is not supported by this modulus");
    }

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1u;
        for (; (j & bit) != 0; bit >>= 1u) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1u) {
        u32 wlen = pow_mod(ROOT, (MOD - 1) / len, MOD);
        if (invert) {
            wlen = pow_mod(wlen, MOD - 2, MOD);
        }
        const std::size_t half = len >> 1u;
        const long long blocks = static_cast<long long>(n / len);

        auto stage = [&](long long block) {
            const std::size_t base = static_cast<std::size_t>(block) * len;
            u32 w = 1;
            for (std::size_t j = 0; j < half; ++j) {
                const u32 u = a[base + j];
                const u32 v = mul_mod(a[base + j + half], w, MOD);
                a[base + j] = add_mod(u, v, MOD);
                a[base + j + half] = sub_mod(u, v, MOD);
                w = mul_mod(w, wlen, MOD);
            }
        };

        if (dynamic_schedule) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
            for (long long block = 0; block < blocks; ++block) {
                stage(block);
            }
        } else {
#pragma omp parallel for num_threads(threads) schedule(static)
            for (long long block = 0; block < blocks; ++block) {
                stage(block);
            }
        }
    }

    if (invert) {
        const u32 inv_n = pow_mod(static_cast<u32>(n), MOD - 2, MOD);
        const long long total = static_cast<long long>(n);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (long long i = 0; i < total; ++i) {
            a[static_cast<std::size_t>(i)] = mul_mod(a[static_cast<std::size_t>(i)], inv_n, MOD);
        }
    }
}

template <u32 MOD, u32 ROOT>
std::vector<u32> convolution_ntt(const std::vector<u32>& a,
                                 const std::vector<u32>& b,
                                 int threads,
                                 bool dynamic_schedule) {
    const std::size_t result_size = a.size() + b.size() - 1;
    const std::size_t ntt_size = ceil_pow2(result_size);
    std::vector<u32> fa(ntt_size, 0);
    std::vector<u32> fb(ntt_size, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        fa[i] = a[i] % MOD;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        fb[i] = b[i] % MOD;
    }

    ntt<MOD, ROOT>(fa, false, threads, dynamic_schedule);
    ntt<MOD, ROOT>(fb, false, threads, dynamic_schedule);

    const long long total = static_cast<long long>(ntt_size);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long long i = 0; i < total; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        fa[idx] = mul_mod(fa[idx], fb[idx], MOD);
    }

    ntt<MOD, ROOT>(fa, true, threads, dynamic_schedule);
    fa.resize(result_size);
    return fa;
}

std::vector<u32> convolution_crt(const std::vector<u32>& a,
                                 const std::vector<u32>& b,
                                 int threads,
                                 bool dynamic_schedule) {
    std::vector<u32> r1;
    std::vector<u32> r2;
    std::vector<u32> r3;

    const int outer_threads = std::min(3, std::max(1, threads));
    const int inner_threads = std::max(1, threads / outer_threads);

    omp_set_nested(1);
#pragma omp parallel sections num_threads(outer_threads)
    {
#pragma omp section
        { r1 = convolution_ntt<MOD_A, ROOT_A>(a, b, inner_threads, dynamic_schedule); }
#pragma omp section
        { r2 = convolution_ntt<MOD_B, ROOT_B>(a, b, inner_threads, dynamic_schedule); }
#pragma omp section
        { r3 = convolution_ntt<MOD_C, ROOT_C>(a, b, inner_threads, dynamic_schedule); }
    }

    const u64 inv_m1_mod_m2 = inverse_mod_u64(MOD_A, MOD_B);
    const u64 m1m2_mod_m3 = (static_cast<u64>(MOD_A) * MOD_B) % MOD_C;
    const u64 inv_m1m2_mod_m3 = inverse_mod_u64(m1m2_mod_m3, MOD_C);
    const u64 m1_mod_target = MOD_A % TARGET_MOD;
    const u64 m1m2_mod_target = (static_cast<u64>(MOD_A) * MOD_B) % TARGET_MOD;

    std::vector<u32> out(r1.size());
    const long long total = static_cast<long long>(out.size());
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long long i = 0; i < total; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const u64 x1 = r1[idx];
        const u64 t2 = ((r2[idx] + MOD_B - (x1 % MOD_B)) % MOD_B) * inv_m1_mod_m2 % MOD_B;
        const u64 x12_mod_m3 = (x1 % MOD_C + static_cast<u64>(MOD_A % MOD_C) * (t2 % MOD_C)) % MOD_C;
        const u64 t3 = ((r3[idx] + MOD_C - x12_mod_m3) % MOD_C) * inv_m1m2_mod_m3 % MOD_C;
        const u64 value = (x1 % TARGET_MOD + m1_mod_target * (t2 % TARGET_MOD) +
                           m1m2_mod_target * (t3 % TARGET_MOD)) %
                          TARGET_MOD;
        out[idx] = static_cast<u32>(value);
    }
    return out;
}

struct Options {
    std::string mode = "all";
    int min_exp = 10;
    int max_exp = 18;
    int direct_limit_exp = 13;
    int repeats = 3;
    u32 max_coeff = 1024;
    std::string csv = "results/bench.csv";
    std::vector<int> thread_counts = {1, 2, 4, 8};
};

std::vector<int> parse_threads(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::max(1, std::stoi(item)));
        }
    }
    return values.empty() ? std::vector<int>{1} : values;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--mode") {
            opt.mode = need_value("--mode");
        } else if (arg == "--min-exp") {
            opt.min_exp = std::stoi(need_value("--min-exp"));
        } else if (arg == "--max-exp") {
            opt.max_exp = std::stoi(need_value("--max-exp"));
        } else if (arg == "--direct-limit-exp") {
            opt.direct_limit_exp = std::stoi(need_value("--direct-limit-exp"));
        } else if (arg == "--repeats") {
            opt.repeats = std::max(1, std::stoi(need_value("--repeats")));
        } else if (arg == "--threads") {
            opt.thread_counts = parse_threads(need_value("--threads"));
        } else if (arg == "--csv") {
            opt.csv = need_value("--csv");
        } else if (arg == "--max-coeff") {
            opt.max_coeff = static_cast<u32>(std::stoul(need_value("--max-coeff")));
        } else if (arg == "--help") {
            std::cout << "Usage: ntt_bench [--mode all|direct|ntt|crt] [--threads 1,2,4,8]\n"
                      << "                 [--min-exp 10] [--max-exp 18] [--direct-limit-exp 13]\n"
                      << "                 [--repeats 3] [--csv results/bench.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opt;
}

template <class F>
std::pair<double, u64> measure_best(int repeats, F&& fn) {
    double best = std::numeric_limits<double>::infinity();
    u64 best_checksum = 0;
    for (int r = 0; r < repeats; ++r) {
        Timer timer;
        std::vector<u32> out = fn();
        const double ms = timer.elapsed_ms();
        if (ms < best) {
            best = ms;
            best_checksum = checksum(out);
        }
    }
    return {best, best_checksum};
}

void emit(std::ofstream& csv,
          const std::string& problem,
          const std::string& variant,
          std::size_t n,
          int threads,
          const std::string& schedule,
          const std::string& simd,
          const std::string& alignment,
          double ms,
          u64 hash) {
    csv << problem << ',' << variant << ',' << n << ',' << threads << ',' << schedule << ',' << simd << ','
        << alignment << ',' << std::fixed << std::setprecision(4) << ms << ',' << hash << '\n';
    std::cout << std::setw(8) << problem << "  " << std::setw(18) << variant << "  n=" << std::setw(8) << n
              << "  t=" << std::setw(2) << threads << "  " << std::setw(10) << ms << " ms\n";
}

void run_direct(const Options& opt, std::ofstream& csv, std::size_t n, const std::vector<u32>& a,
                const std::vector<u32>& b) {
    if (n > (std::size_t{1} << opt.direct_limit_exp)) {
        return;
    }
    auto scalar = measure_best(opt.repeats, [&] { return direct_scalar<TARGET_MOD>(a, b); });
    emit(csv, "direct", "scalar", n, 1, "none", "none", "none", scalar.first, scalar.second);

    auto avx_unaligned = measure_best(opt.repeats, [&] { return direct_avx2<TARGET_MOD>(a, b, false); });
    emit(csv, "direct", "manual-avx2", n, 1, "none", "avx2", "unaligned", avx_unaligned.first,
         avx_unaligned.second);

    auto avx_aligned = measure_best(opt.repeats, [&] { return direct_avx2<TARGET_MOD>(a, b, true); });
    emit(csv, "direct", "manual-avx2", n, 1, "none", "avx2", "prefer-aligned", avx_aligned.first,
         avx_aligned.second);

    for (int threads : opt.thread_counts) {
        auto omp_static = measure_best(opt.repeats, [&] { return direct_omp<TARGET_MOD>(a, b, threads, false); });
        emit(csv, "direct", "openmp-output", n, threads, "static", "auto", "none", omp_static.first,
             omp_static.second);

        auto omp_dynamic = measure_best(opt.repeats, [&] { return direct_omp<TARGET_MOD>(a, b, threads, true); });
        emit(csv, "direct", "openmp-output", n, threads, "dynamic", "auto", "none", omp_dynamic.first,
             omp_dynamic.second);

#if defined(HAVE_PTHREAD_IMPL)
        auto pthread_static = measure_best(opt.repeats, [&] { return direct_pthread<TARGET_MOD>(a, b, threads); });
        emit(csv, "direct", "pthread-output", n, threads, "static", "none", "none", pthread_static.first,
             pthread_static.second);
#endif
    }
}

void run_ntt(const Options& opt, std::ofstream& csv, std::size_t n, const std::vector<u32>& a,
             const std::vector<u32>& b) {
    auto scalar = measure_best(opt.repeats, [&] { return convolution_ntt<MOD_A, ROOT_A>(a, b, 1, false); });
    emit(csv, "ntt", "single-mod-scalar", n, 1, "static", "auto", "none", scalar.first, scalar.second);

    for (int threads : opt.thread_counts) {
        auto omp_static = measure_best(opt.repeats, [&] { return convolution_ntt<MOD_A, ROOT_A>(a, b, threads, false); });
        emit(csv, "ntt", "single-mod-openmp", n, threads, "static", "auto", "none", omp_static.first,
             omp_static.second);

        auto omp_dynamic = measure_best(opt.repeats, [&] { return convolution_ntt<MOD_A, ROOT_A>(a, b, threads, true); });
        emit(csv, "ntt", "single-mod-openmp", n, threads, "dynamic", "auto", "none", omp_dynamic.first,
             omp_dynamic.second);
    }
}

void run_crt(const Options& opt, std::ofstream& csv, std::size_t n, const std::vector<u32>& a,
             const std::vector<u32>& b) {
    for (int threads : opt.thread_counts) {
        auto task_static = measure_best(opt.repeats, [&] { return convolution_crt(a, b, threads, false); });
        emit(csv, "crt", "three-mod-task", n, threads, "static", "auto", "none", task_static.first,
             task_static.second);

        auto task_dynamic = measure_best(opt.repeats, [&] { return convolution_crt(a, b, threads, true); });
        emit(csv, "crt", "three-mod-task", n, threads, "dynamic", "auto", "none", task_dynamic.first,
             task_dynamic.second);
    }
}

}  // namespace bench

int main(int argc, char** argv) {
    try {
        const bench::Options opt = bench::parse_args(argc, argv);
        std::ofstream csv(opt.csv);
        if (!csv) {
            throw std::runtime_error("cannot open csv file: " + opt.csv);
        }
        csv << "problem,variant,n,threads,schedule,simd,alignment,ms,checksum\n";

        std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#if defined(__AVX2__)
        std::cout << "AVX2: enabled at compile time\n";
#else
        std::cout << "AVX2: disabled at compile time, SIMD path falls back to scalar\n";
#endif

        for (int exp = opt.min_exp; exp <= opt.max_exp; ++exp) {
            const std::size_t n = std::size_t{1} << exp;
            const auto a = bench::random_poly(n, opt.max_coeff, 20260524u + static_cast<bench::u32>(exp));
            const auto b = bench::random_poly(n, opt.max_coeff, 20260525u + static_cast<bench::u32>(exp));

            if (opt.mode == "all" || opt.mode == "direct") {
                bench::run_direct(opt, csv, n, a, b);
            }
            if (opt.mode == "all" || opt.mode == "ntt") {
                bench::run_ntt(opt, csv, n, a, b);
            }
            if (opt.mode == "all" || opt.mode == "crt") {
                bench::run_crt(opt, csv, n, a, b);
            }
        }
        std::cout << "CSV written to " << opt.csv << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
