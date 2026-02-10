#!/usr/bin/env python3
"""
=============================================================
Edge ML - Inferencia no Raspberry Pi Zero 2 W
Classificacao de Falhas em Equipamentos Rotativos
=============================================================

Uso:
    python3 inference.py                    # Roda tudo
    python3 inference.py --mode test        # So dados de teste
    python3 inference.py --mode synthetic   # So dados sinteticos (cGAN)
    python3 inference.py --mode benchmark   # So benchmark de latencia
    python3 inference.py --model model_quantized_int8.tflite
"""

import numpy as np
import json
import csv
import time
import argparse
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLS = ['Vibration_X', 'Vibration_Y', 'Vibration_Z',
                'Acoustic_Level', 'Temperature']


# -- Helpers ---------------------------------------------------
def load_scaler(path):
    with open(path) as f:
        p = json.load(f)
    return np.array(p['mean'], dtype=np.float32), np.array(p['scale'], dtype=np.float32)

def load_labels(path):
    with open(path) as f:
        m = json.load(f)
    return {int(k): v for k, v in m.items()}

def load_csv(path):
    X, y = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            X.append([float(row[c]) for c in FEATURE_COLS])
            y.append(int(float(row['Fault_Type_Encoded'])))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def standardize(X, mean, scale):
    return (X - mean) / scale

def get_interpreter(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError:
                print("[ERRO] Instale: pip install ai-edge-litert")
                sys.exit(1)

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    kb = os.path.getsize(model_path) / 1024
    print(f"  Modelo:  {os.path.basename(model_path)} ({kb:.1f} KB)")
    print(f"  Input:   {inp[0]['shape']}  dtype={inp[0]['dtype']}")
    print(f"  Output:  {out[0]['shape']}  dtype={out[0]['dtype']}")
    return interp, inp, out

def infer_one(interp, inp, out, sample):
    interp.set_tensor(inp[0]['index'], sample.reshape(1, -1).astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out[0]['index'])[0]

def infer_batch(interp, inp, out, X):
    probas, times = [], []
    for i in range(len(X)):
        t0 = time.perf_counter()
        p = infer_one(interp, inp, out, X[i])
        times.append(time.perf_counter() - t0)
        probas.append(p)
    return np.array(probas), np.array(times)


# -- Metricas (sem sklearn) ------------------------------------
def conf_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def print_cm(cm, labels):
    n = len(labels)
    w = max(len(labels[i]) for i in range(n))
    hdr = " " * (w + 4) + "".join(f"{labels[i]:>14s}" for i in range(n)) + "   Recall"
    print(hdr)
    print("-" * len(hdr))
    for i in range(n):
        total = cm[i].sum()
        rec = cm[i][i] / total if total else 0
        row = f"  {labels[i]:>{w}s} |" + "".join(f"{cm[i][j]:>14d}" for j in range(n))
        print(f"{row}   {rec:.4f}")
    print("-" * len(hdr))

def classification_report(y_true, y_pred, labels, n):
    print(f"\n  {'Classe':<18s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Sup':>8s}")
    print("  " + "-" * 52)
    precs, recs, f1s, sups = [], [], [], []
    for c in range(n):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        sup = int((y_true == c).sum())
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        precs.append(p); recs.append(r); f1s.append(f); sups.append(sup)
        print(f"  {labels[c]:<18s} {p:>8.4f} {r:>8.4f} {f:>8.4f} {sup:>8d}")
    total = sum(sups)
    print("  " + "-" * 52)
    wp = sum(p*s for p,s in zip(precs,sups))/total
    wr = sum(r*s for r,s in zip(recs,sups))/total
    wf = sum(f*s for f,s in zip(f1s,sups))/total
    print(f"  {'Weighted Avg':<18s} {wp:>8.4f} {wr:>8.4f} {wf:>8.4f} {total:>8d}")
    acc = float((y_true == y_pred).sum()) / len(y_true)
    print(f"\n  Accuracy: {acc:.4f}  ({(y_true == y_pred).sum()}/{len(y_true)})")
    return acc

def print_timing(times, tag=""):
    print(f"\n  Timing {tag}:")
    print(f"    Total:      {times.sum():.4f} s")
    print(f"    Media:      {times.mean()*1000:.3f} ms/amostra")
    print(f"    Mediana:    {np.median(times)*1000:.3f} ms")
    print(f"    Std:        {times.std()*1000:.3f} ms")
    print(f"    P95:        {np.percentile(times,95)*1000:.3f} ms")
    print(f"    Min/Max:    {times.min()*1000:.3f} / {times.max()*1000:.3f} ms")
    print(f"    Throughput: {1.0/times.mean():.0f} amostras/s")


# -- Modos de execucao -----------------------------------------
def run_test(interp, inp, out, mean, scl, labels, nc):
    path = os.path.join(BASE_DIR, "test_data.csv")
    if not os.path.exists(path):
        print(f"  [ERRO] {path} nao encontrado!"); return

    print("\n" + "=" * 60)
    print("  DADOS DE TESTE (mesmos do Colab)")
    print("=" * 60)
    X, y = load_csv(path)
    Xs = standardize(X, mean, scl)
    print(f"  Amostras: {len(X)}")

    probas, times = infer_batch(interp, inp, out, Xs)
    yp = np.argmax(probas, axis=1)

    cm = conf_matrix(y, yp, nc)
    print("\n  Matriz de Confusao:")
    print_cm(cm, labels)
    acc = classification_report(y, yp, labels, nc)
    print_timing(times, "(teste)")

    res = {"timestamp": datetime.now().isoformat(), "mode": "test",
           "n_samples": len(X), "accuracy": round(acc, 6),
           "avg_ms": round(float(times.mean()*1000), 3),
           "throughput": round(float(1/times.mean()), 1)}
    with open(os.path.join(BASE_DIR, "results_test.json"), 'w') as f:
        json.dump(res, f, indent=2)
    print("  Salvo: results_test.json")


def run_synthetic(interp, inp, out, mean, scl, labels, nc):
    path = os.path.join(BASE_DIR, "synthetic_data.csv")
    if not os.path.exists(path):
        print(f"  [ERRO] {path} nao encontrado!"); return

    print("\n" + "=" * 60)
    print("  DADOS SINTETICOS (cGAN)")
    print("=" * 60)
    X, y = load_csv(path)
    Xs = standardize(X, mean, scl)
    print(f"  Amostras: {len(X)}")

    probas, times = infer_batch(interp, inp, out, Xs)
    yp = np.argmax(probas, axis=1)

    cm = conf_matrix(y, yp, nc)
    print("\n  Matriz de Confusao:")
    print_cm(cm, labels)
    acc = classification_report(y, yp, labels, nc)
    print_timing(times, "(sinteticos)")

    res = {"timestamp": datetime.now().isoformat(), "mode": "synthetic",
           "n_samples": len(X), "accuracy": round(acc, 6),
           "avg_ms": round(float(times.mean()*1000), 3)}
    with open(os.path.join(BASE_DIR, "results_synthetic.json"), 'w') as f:
        json.dump(res, f, indent=2)
    print("  Salvo: results_synthetic.json")


def run_benchmark(interp, inp, out):
    print("\n" + "=" * 60)
    print("  BENCHMARK DE PERFORMANCE")
    print("=" * 60)

    N = 1000
    X_rand = np.random.randn(N, len(FEATURE_COLS)).astype(np.float32)

    # Warmup
    for _ in range(10):
        infer_one(interp, inp, out, X_rand[0])

    times = []
    t_total_start = time.perf_counter()
    for i in range(N):
        t0 = time.perf_counter()
        infer_one(interp, inp, out, X_rand[i])
        times.append(time.perf_counter() - t0)
    t_total = time.perf_counter() - t_total_start
    times = np.array(times)

    print(f"\n  {N} inferencias:")
    print(f"    Media:      {times.mean()*1000:.3f} ms")
    print(f"    Mediana:    {np.median(times)*1000:.3f} ms")
    print(f"    Std:        {times.std()*1000:.3f} ms")
    print(f"    P5/P95:     {np.percentile(times,5)*1000:.3f} / {np.percentile(times,95)*1000:.3f} ms")
    print(f"    P99:        {np.percentile(times,99)*1000:.3f} ms")
    print(f"    Min/Max:    {times.min()*1000:.3f} / {times.max()*1000:.3f} ms")
    print(f"    Throughput: {N/t_total:.0f} amostras/s")

    # Info do sistema
    print("\n  Sistema:")
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if any(k in line.lower() for k in ['model name', 'hardware']):
                    print(f"    {line.strip()}"); break
        with open('/proc/meminfo') as f:
            for line in f:
                if any(k in line.lower() for k in ['memtotal', 'memavailable']):
                    print(f"    {line.strip()}")
    except: pass

    res = {"timestamp": datetime.now().isoformat(), "n_inferences": N,
           "mean_ms": round(float(times.mean()*1000), 3),
           "median_ms": round(float(np.median(times)*1000), 3),
           "p95_ms": round(float(np.percentile(times,95)*1000), 3),
           "throughput": round(float(N/t_total), 1)}
    with open(os.path.join(BASE_DIR, "results_benchmark.json"), 'w') as f:
        json.dump(res, f, indent=2)
    print("  Salvo: results_benchmark.json")


# -- Main ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Edge ML - RPi Inference')
    parser.add_argument('--model', default='model_float32.tflite')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'test', 'synthetic', 'benchmark'])
    args = parser.parse_args()

    print("=" * 60)
    print("  Edge ML - Classificacao de Falhas em Equipamentos")
    print("  Raspberry Pi Zero 2 W + TFLite Runtime")
    print("=" * 60)

    mean, scl = load_scaler(os.path.join(BASE_DIR, "scaler_params.json"))
    labels = load_labels(os.path.join(BASE_DIR, "label_mapping.json"))
    nc = len(labels)
    print(f"\n  Classes: {labels}")

    mp = args.model if os.path.isabs(args.model) else os.path.join(BASE_DIR, args.model)
    interp, inp_d, out_d = get_interpreter(mp)

    if args.mode in ('all', 'test'):
        run_test(interp, inp_d, out_d, mean, scl, labels, nc)
    if args.mode in ('all', 'synthetic'):
        run_synthetic(interp, inp_d, out_d, mean, scl, labels, nc)
    if args.mode in ('all', 'benchmark'):
        run_benchmark(interp, inp_d, out_d)

    print("\n" + "=" * 60)
    print("  Concluido!")
    print("=" * 60)


if __name__ == '__main__':
    main()
