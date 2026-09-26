#include "Finetuner.h"
#include "Effects.h"
#include "Fft.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <thread>

namespace invsynth {

namespace {

constexpr int kFftSizes[3] = {512, 1024, 2048};
constexpr double kEps = 1e-7;
using Obj = std::array<double, 2>;

// ------------------------------------------------------------- features
struct Mag { int frames = 0, bins = 0; std::vector<float> v; };

Mag magnitude(const float* x, int n, RealFft& fft) {
    const int N = fft.size(), hop = N / 4;
    std::vector<float> padded;
    if (n < N) { padded.assign((size_t)N, 0.f); std::copy(x, x + n, padded.begin()); x = padded.data(); n = N; }
    Mag m;
    m.frames = 1 + (n - N) / hop;
    m.bins = N / 2 + 1;
    m.v.resize((size_t)m.frames * m.bins);
    for (int f = 0; f < m.frames; ++f) fft.spectrum(x + (size_t)f * hop, &m.v[(size_t)f * m.bins], false);
    return m;
}

std::vector<double> rmsEnvDb(const float* x, int n) {
    const int frames = std::max(1, n / kHop);
    std::vector<double> e((size_t)frames);
    double mx = -1e300;
    for (int f = 0; f < frames; ++f) {
        double acc = 0.0;
        for (int i = 0; i < kHop; ++i) {
            const size_t k = (size_t)f * kHop + i;
            const double v = k < (size_t)n ? x[k] : 0.0;
            acc += v * v;
        }
        e[(size_t)f] = 10.0 * std::log10(acc / kHop + 1e-12);
        mx = std::max(mx, e[(size_t)f]);
    }
    for (double& v : e) v = std::max(v - mx, -60.0);
    return e;
}

struct Target {
    std::vector<float> x;
    Mag mags[3];
    double norms[3] = {};
    std::vector<double> env;

    explicit Target(std::vector<float> seg) : x(std::move(seg)) {
        peakNormalizeInPlace(x.data(), (int)x.size());
        for (int i = 0; i < 3; ++i) {
            RealFft fft(kFftSizes[i]);
            mags[i] = magnitude(x.data(), (int)x.size(), fft);
            double s = 0; for (float v : mags[i].v) s += (double)v * v;
            norms[i] = std::sqrt(s);
        }
        env = rmsEnvDb(x.data(), (int)x.size());
    }

    Obj objectives(const std::vector<float>& y, RealFft* ffts) const {
        double f1 = 0.0;
        for (int i = 0; i < 3; ++i) {
            const Mag r = magnitude(y.data(), (int)y.size(), ffts[i]);
            const auto& T = mags[i].v;
            const size_t cnt = std::min(T.size(), r.v.size());
            double diff = 0.0, lm = 0.0;
            for (size_t k = 0; k < cnt; ++k) {
                const double d = (double)T[k] - r.v[k];
                diff += d * d;
                lm += std::abs(std::log(T[k] + kEps) - std::log(r.v[k] + kEps));
            }
            f1 += std::sqrt(diff) / (norms[i] + kEps) + 0.1 * lm / (double)cnt;
        }
        f1 /= 3.0;
        const auto e = rmsEnvDb(y.data(), (int)y.size());
        double f2 = 0.0;
        const size_t m = std::min(env.size(), e.size());
        for (size_t k = 0; k < m; ++k) f2 += std::abs(env[k] - e[k]);
        f2 = f2 / (double)std::max<size_t>(m, 1) / 60.0;
        return {f1, f2};
    }
};

double scalar(const Obj& o) { return o[0] + 0.5 * o[1]; }

// ------------------------------------------------------------- NSGA-II
bool dominates(const Obj& a, const Obj& b) {
    return a[0] <= b[0] && a[1] <= b[1] && (a[0] < b[0] || a[1] < b[1]);
}

std::vector<std::vector<int>> nondominatedSort(const std::vector<Obj>& obj, std::vector<int>& rank) {
    const int n = (int)obj.size();
    std::vector<std::vector<int>> S((size_t)n), fronts(1);
    std::vector<int> cnt((size_t)n, 0);
    rank.assign((size_t)n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dominates(obj[(size_t)i], obj[(size_t)j])) S[(size_t)i].push_back(j);
            else if (dominates(obj[(size_t)j], obj[(size_t)i])) ++cnt[(size_t)i];
        }
        if (cnt[(size_t)i] == 0) fronts[0].push_back(i);
    }
    size_t k = 0;
    while (!fronts[k].empty()) {
        std::vector<int> nxt;
        for (int i : fronts[k])
            for (int j : S[(size_t)i])
                if (--cnt[(size_t)j] == 0) { rank[(size_t)j] = (int)k + 1; nxt.push_back(j); }
        ++k;
        fronts.push_back(std::move(nxt));
    }
    fronts.pop_back();
    return fronts;
}

std::vector<double> crowding(const std::vector<Obj>& obj, const std::vector<int>& front) {
    const size_t n = front.size();
    std::vector<double> d(n, 0.0);
    if (n <= 2) { std::fill(d.begin(), d.end(), std::numeric_limits<double>::infinity()); return d; }
    std::vector<size_t> o(n);
    for (int m = 0; m < 2; ++m) {
        for (size_t i = 0; i < n; ++i) o[i] = i;
        std::sort(o.begin(), o.end(), [&](size_t a, size_t b) {
            return obj[(size_t)front[a]][(size_t)m] < obj[(size_t)front[b]][(size_t)m]; });
        d[o.front()] = d[o.back()] = std::numeric_limits<double>::infinity();
        const double span = obj[(size_t)front[o.back()]][(size_t)m] - obj[(size_t)front[o.front()]][(size_t)m] + 1e-12;
        for (size_t i = 1; i + 1 < n; ++i)
            d[o[i]] += (obj[(size_t)front[o[i + 1]]][(size_t)m] - obj[(size_t)front[o[i - 1]]][(size_t)m]) / span;
    }
    return d;
}

bool isCategorical(int i) { return paramTable()[(size_t)i].numClasses > 0; }

double timeToNorm(double t) {
    return std::clamp(std::log(std::max(t, 1e-3) / 0.001) / std::log(10000.0), 0.0, 1.0);
}

} // namespace

// ------------------------------------------------------ duration matching
ParamVector matchDuration(const ParamVector& p0, const float* note, int n) {
    ParamVector p = p0;
    const auto env = rmsEnvDb(note, n);
    const double fps = kFramesPerSec, D = (double)n / kSR;
    const int T = (int)env.size();
    int tp = 0;
    while (tp < T && env[(size_t)tp] < -1.0) ++tp;
    if (tp == T) tp = 0;
    p[kAEnvA] = (float)timeToNorm(tp / fps);
    int tg = -1, te = -1;
    for (int t = T - 1; t >= 0 && tg < 0; --t) if (env[(size_t)t] > -12.0) tg = t;
    for (int t = T - 1; t >= 0 && te < 0; --t) if (env[(size_t)t] > -59.0) te = t;
    if (tg < 0) tg = tp;
    if (te < 0) te = T - 1;
    if (te >= T - 2 && tg >= T - 3) {
        p[kGate] = 1.f;
    } else {
        p[kGate] = (float)std::clamp((tg / fps / D - 0.05) / 0.95, 0.0, 1.0);
        const int len = te - tg + 1;
        if (len >= 3) {                                  // least-squares dB/s slope
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (int i = 0; i < len; ++i) {
                const double xx = i / fps, yy = env[(size_t)(tg + i)];
                sx += xx; sy += yy; sxx += xx * xx; sxy += xx * yy;
            }
            const double slope = (len * sxy - sx * sy) / (len * sxx - sx * sx + 1e-12);
            if (slope < -1e-3) p[kAEnvR] = (float)timeToNorm(5.0 * 8.686 / -slope);
        }
    }
    if (tg > tp + (int)(0.2 * fps)) {
        std::vector<double> seg(env.begin() + tp, env.begin() + tg + 1);
        std::nth_element(seg.begin(), seg.begin() + seg.size() / 2, seg.end());
        double med = seg[seg.size() / 2];
        if (seg.size() % 2 == 0) {       // numpy median averages the two middle values
            const double lo = *std::max_element(seg.begin(), seg.begin() + seg.size() / 2);
            med = 0.5 * (med + lo);
        }
        p[kAEnvS] = (float)std::clamp(std::pow(10.0, med / 20.0), 0.0, 1.0);
    }
    return p;
}

// ---------------------------------------------------------------- driver
FinetuneResult finetune(const std::vector<float>& x, const ParamVector& p0,
                        const AnalysisWindow& w, const FinetuneConfig& cfg,
                        const ProgressFn& progress, const std::atomic<bool>* cancel) {
    FinetuneResult result;
    result.params = p0;
    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    const int noteStart = w.start;
    const int nFull = (int)x.size() - noteStart;
    const float* note = x.data() + noteStart;
    const int cap = (int)(cfg.maxCompareSec * kSR);
    const int sel = std::min(w.length, nFull), full = std::min(nFull, cap);
    int ends[3] = {sel, sel, sel};
    if (cfg.mode == WindowMode::Full) ends[0] = ends[1] = ends[2] = full;
    else if (cfg.mode == WindowMode::Progressive) { ends[1] = std::min(2 * sel, full); ends[2] = full; }

    const int threads = cfg.threads > 0 ? cfg.threads
                      : (int)std::max(1u, std::thread::hardware_concurrency() - 1);
    std::vector<int> contIdx;
    for (int i = 0; i < kNumParams; ++i) if (!isCategorical(i)) contIdx.push_back(i);

    // ---- initial population
    std::vector<ParamVector> pop;
    pop.push_back(p0);
    pop.push_back(matchDuration(p0, note, std::min(nFull, cap)));
    while ((int)pop.size() < cfg.population) {
        ParamVector b = pop[(size_t)(rng() % 2)];
        for (int i : contIdx) b[(size_t)i] = (float)std::clamp(b[(size_t)i] + 0.05 * N01(rng), 0.0, 1.0);
        for (int i = 0; i < kNumParams; ++i)
            if (isCategorical(i) && U(rng) < 0.1) {
                const int k = paramTable()[(size_t)i].numClasses;
                b[(size_t)i] = (float)(((double)(rng() % (uint64_t)k) + 0.5) / k);
            }
        pop.push_back(b);
    }

    std::unique_ptr<Target> target;
    auto evaluate = [&](const std::vector<ParamVector>& P) {
        std::vector<Obj> out(P.size());
        std::atomic<size_t> next{0};
        auto worker = [&] {
            RealFft ffts[3] = {RealFft(512), RealFft(1024), RealFft(2048)};
            for (size_t i; (i = next++) < P.size();) {
                if (cancel && cancel->load()) { out[i] = {10, 10}; continue; }
                auto y = renderNote(P[i], (int)target->x.size(), gateSamples(P[i], nFull), kSR, true);
                float pk = 0; for (float v : y) pk = std::max(pk, std::abs(v));
                out[i] = (pk < 1e-6f || !std::isfinite(pk)) ? Obj{10, 10} : target->objectives(y, ffts);
            }
        };
        std::vector<std::thread> ts;
        for (int t = 1; t < threads; ++t) ts.emplace_back(worker);
        worker();
        for (auto& t : ts) t.join();
        return out;
    };

    auto sbx = [&](float& a, float& b) {
        const double u = U(rng), eta = 15.0;
        const double beta = u <= 0.5 ? std::pow(2 * u, 1 / (eta + 1)) : std::pow(1 / (2 * (1 - u)), 1 / (eta + 1));
        const double c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b), c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b);
        a = (float)std::clamp(c1, 0.0, 1.0); b = (float)std::clamp(c2, 0.0, 1.0);
    };
    auto polyMut = [&](float& v) {
        const double u = U(rng), eta = 20.0;
        const double d = u < 0.5 ? std::pow(2 * u, 1 / (eta + 1)) - 1 : 1 - std::pow(2 * (1 - u), 1 / (eta + 1));
        v = (float)std::clamp(v + d, 0.0, 1.0);
    };

    std::vector<Obj> obj;
    ParamVector best = p0;
    double bestL = 1e18;
    int stall = 0, stage = -1;
    const int perStage = std::max(1, cfg.generations / 3);

    for (int g = 0; g < cfg.generations; ++g) {
        if (cancel && cancel->load()) { result.cancelled = true; break; }
        const int s = std::min(g / perStage, 2);
        if (s != stage) {
            stage = s;
            target = std::make_unique<Target>(std::vector<float>(note, note + ends[s]));
            if (s > 0) pop.back() = matchDuration(best, note, ends[s]);
            obj = evaluate(pop);
            bestL = 1e18;
        }
        std::vector<int> rank;
        auto fronts = nondominatedSort(obj, rank);
        std::vector<double> crowd(pop.size(), 0.0);
        for (auto& f : fronts) { auto c = crowding(obj, f); for (size_t i = 0; i < f.size(); ++i) crowd[(size_t)f[i]] = c[i]; }

        auto tour = [&]() -> const ParamVector& {
            const size_t i = rng() % pop.size(), j = rng() % pop.size();
            if (rank[i] != rank[j]) return rank[i] < rank[j] ? pop[i] : pop[j];
            return crowd[i] > crowd[j] ? pop[i] : pop[j];
        };
        std::vector<ParamVector> kids;
        while (kids.size() < pop.size()) {
            ParamVector c1 = tour(), c2 = tour();
            if (U(rng) < 0.9) for (int i : contIdx) sbx(c1[(size_t)i], c2[(size_t)i]);
            for (int i = 0; i < kNumParams; ++i) {
                if (!isCategorical(i)) continue;
                if (U(rng) < 0.5) std::swap(c1[(size_t)i], c2[(size_t)i]);
                const int k = paramTable()[(size_t)i].numClasses;
                for (auto* c : {&c1, &c2})
                    if (U(rng) < 0.05) (*c)[(size_t)i] = (float)(((double)(rng() % (uint64_t)k) + 0.5) / k);
            }
            const double pm = 2.0 / contIdx.size();
            for (auto* c : {&c1, &c2}) {
                for (int i : contIdx) if (U(rng) < pm) polyMut((*c)[(size_t)i]);
                kids.push_back(*c);
            }
        }
        kids.resize(pop.size());
        auto kobj = evaluate(kids);

        std::vector<ParamVector> allP = pop;  allP.insert(allP.end(), kids.begin(), kids.end());
        std::vector<Obj> allO = obj;          allO.insert(allO.end(), kobj.begin(), kobj.end());
        std::vector<int> r2;
        auto f2 = nondominatedSort(allO, r2);
        std::vector<int> selIdx;
        for (auto& f : f2) {
            if (selIdx.size() + f.size() <= pop.size()) { selIdx.insert(selIdx.end(), f.begin(), f.end()); continue; }
            auto cd = crowding(allO, f);
            std::vector<size_t> o(f.size());
            for (size_t i = 0; i < o.size(); ++i) o[i] = i;
            std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return cd[a] > cd[b]; });
            for (size_t i = 0; selIdx.size() < pop.size(); ++i) selIdx.push_back(f[o[i]]);
            break;
        }
        for (size_t i = 0; i < selIdx.size(); ++i) { pop[i] = allP[(size_t)selIdx[i]]; obj[i] = allO[(size_t)selIdx[i]]; }

        size_t bi = 0;
        for (size_t i = 1; i < obj.size(); ++i) if (scalar(obj[i]) < scalar(obj[bi])) bi = i;
        if (scalar(obj[bi]) < bestL - 1e-4) { bestL = scalar(obj[bi]); best = pop[bi]; stall = 0; }
        else ++stall;
        result.generationsRun = g + 1;
        if (progress) progress(g, cfg.generations, bestL, (double)ends[s] / kSR);
        if (s == 2 && (bestL < cfg.tolerance || stall >= cfg.patience)) break;
    }
    result.params = best;
    result.loss = bestL;
    return result;
}

} // namespace invsynth
