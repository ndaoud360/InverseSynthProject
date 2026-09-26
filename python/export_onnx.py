#!/usr/bin/env python3
"""
Phase 4 - export the predictor to ONNX.

* Inputs:  mel (1, 128, T) float32 with DYNAMIC T,  ctx (1, 3) float32
* Output:  params (1, 41) float32, already assembled (categoricals -> bin centres)
* Feature-extraction constants are written into the model's metadata so the
  C++ host can refuse a model whose front-end does not match its own.

    python export_onnx.py --ckpt checkpoints/best.pt --out inverse_synth.onnx [--int8]
"""
import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from invsynth import features as F, params as P
from invsynth.model import ExportWrapper, InverseSynthNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="inverse_synth.onnx")
    ap.add_argument("--int8", action="store_true", help="also write dynamic-int8 model")
    a = ap.parse_args()

    net = InverseSynthNet()
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    net.eval()
    model = ExportWrapper(net).eval()

    T = 431
    mel = torch.randn(1, F.N_MELS, T) * 0.3 - 0.5
    ctx = torch.tensor([[0.0, 1.0, 5.0 / 60.0]])
    torch.onnx.export(model, (mel, ctx), a.out, input_names=["mel", "ctx"],
                      output_names=["params"], opset_version=17,
                      dynamic_axes={"mel": {2: "time"}}, dynamo=False)

    m = onnx.load(a.out)
    meta = dict(sr=F.SR, n_fft=F.N_FFT, hop=F.HOP, n_mels=F.N_MELS, fmin=F.FMIN,
                fmax=F.FMAX, window_sec=F.WINDOW_SEC, n_params=P.N_PARAMS,
                db_floor=F.DB_FLOOR, param_names=[p.name for p in P.PARAMS])
    onnx.helper.set_model_props(m, {"invsynth_config": json.dumps(meta)})
    onnx.save(m, a.out)

    # --- verify: several lengths, compare against PyTorch -------------------
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    for T in (22, 173, 431, 900):
        mel = (torch.randn(1, F.N_MELS, T) * 0.3 - 0.5)
        with torch.no_grad():
            ref = model(mel, ctx).numpy()
        out = sess.run(None, {"mel": mel.numpy(), "ctx": ctx.numpy()})[0]
        err = np.abs(ref - out)[0, P.CONT_IDX].max()
        print(f"T={T:4d}  max|torch-onnx| = {err:.2e}")
        assert err < 1e-3

    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = a.out.replace(".onnx", "_int8.onnx")
        quantize_dynamic(a.out, q, weight_type=QuantType.QInt8)
        print("wrote", q)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
