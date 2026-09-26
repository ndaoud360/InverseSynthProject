// Headless tool: parity tests against Python + offline analysis without a DAW.
//
//   invsynth_cli render  <params.txt> <n_samples> <gate_samples> <out.f32>
//   invsynth_cli mel     <in.f32> <out.f32>            (features of whole clip)
//   invsynth_cli window  <in.f32>                      (prints start length total)
//   invsynth_cli analyse <in.f32> <model.onnx> <onnxruntime lib> [generations]
//
// .f32 = raw little-endian float32 mono @ 22050 Hz.
#include "../core/Effects.h"
#include "../core/Features.h"
#include "../core/Pipeline.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace invsynth;

static std::vector<float> readF32(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto sz = (size_t)f.tellg();
    std::vector<float> v(sz / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)sz);
    return v;
}
static void writeF32(const char* path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * 4));
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "see source header for usage\n"; return 1; }
    const std::string cmd = argv[1];
    if (cmd == "render" && argc == 6) {
        ParamVector p{};
        std::ifstream f(argv[2]);
        for (auto& v : p) f >> v;
        writeF32(argv[5], renderNote(p, std::atoi(argv[3]), std::atoi(argv[4]), kSR, false));
        return 0;
    }
    if (cmd == "mel" && argc == 4) {
        auto x = readF32(argv[2]);
        auto s = melDb(x.data(), (int)x.size());
        dbToFeature(s);
        writeF32(argv[3], s.v);
        std::cout << s.nMels << " " << s.T << "\n";
        return 0;
    }
    if (cmd == "window" && argc == 3) {
        auto x = readF32(argv[2]);
        peakNormalizeInPlace(x.data(), (int)x.size());
        auto w = selectWindow(x);
        std::cout << w.start << " " << w.length << " " << w.total << "\n";
        return 0;
    }
    if (cmd == "analyse" && argc >= 5) {
        std::string err;
        if (!Predictor::loadRuntime(argv[4], err)) { std::cerr << err << "\n"; return 2; }
        Predictor pred;
        if (!pred.loadModel(argv[3], err)) { std::cerr << err << "\n"; return 3; }
        AnalysisOptions opt;
        if (argc >= 6) opt.finetune.generations = std::atoi(argv[5]);
        auto t0 = std::chrono::steady_clock::now();
        auto r = analyse(readF32(argv[2]), pred, opt, [](int g, int n, double l, double ws) {
            std::fprintf(stderr, "\rgen %d/%d loss %.4f window %.1fs   ", g + 1, n, l, ws);
        });
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "\n%s in %.1fs, loss %.4f, window %.2f+%.2fs\n", r.message.c_str(), secs,
                     r.loss, (double)r.window.start / kSR, (double)r.window.length / kSR);
        for (int i = 0; i < kNumParams; ++i)
            std::printf("%-14s %.4f  %s\n", paramTable()[(size_t)i].id, r.params[(size_t)i],
                        paramToText(i, r.params[(size_t)i]).c_str());
        return r.ok ? 0 : 4;
    }
    std::cerr << "bad arguments\n";
    return 1;
}
