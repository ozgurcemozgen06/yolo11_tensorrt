# TensorRT — What it is, how it works, and how I used it

## 1) What is TensorRT?
NVIDIA TensorRT is a C++/Python SDK that **optimizes trained neural networks** and runs them with a high-performance inference runtime on NVIDIA GPUs. It fuses layers, chooses fast kernel implementations (tactics), and lowers precision (FP16/INT8) while preserving accuracy.

**Why use it?**
- Lower latency (ms/sample) and higher throughput (FPS)
- Lower tail latency (p95)
- Reduced memory bandwidth and power (especially with FP16/INT8)

---

## 2) Where TensorRT fits in the pipeline 

PyTorch (.pt) ──(export)──> ONNX ──(build/optimize)──> TensorRT Engine (.engine) ──(run)──> Inference
(Ultralytics exporter or torch.onnx) (TensorRT builder) (TensorRT runtime)


In this work, I used **Ultralytics’ built-in exporter** to go directly from `.pt` → `.engine` (it performs an ONNX step under the hood).

---

## 3) Key concepts & vocabulary

- **Network / Engine**  
  The *engine* (`.engine` file) is the compiled, hardware-specific, optimized form of the model.

- **Builder & BuilderConfig**  
  The *builder* picks optimization tactics and emits the engine. The *config* controls precision (FP32/FP16/INT8), workspace (memory for tactic search), and optimization profiles.

- **Optimization profile (static vs dynamic shapes)**  
  Engines can be built **static** (fixed shapes, faster to build & simple to use) or **dynamic** (ranges for batch/shape, more flexible but needs a selected profile at runtime).

- **Precision modes**
  - **FP32**: full precision, baseline accuracy/speed.
  - **FP16**: usually ~1.3–2× faster with near-identical accuracy on modern GPUs.
  - **INT8**: fastest; requires **calibration**; small accuracy trade-off is common.

- **Calibration (INT8 only)**
  A short pass over representative images to estimate activation ranges. Better coverage → better accuracy retention.

- **Bindings**
  Pointers to input/output tensors the runtime uses during execution.

---

## 4) Practical workflow been followed

### (A) Export → Build engines
- Start from a trained PyTorch checkpoint (`yolo11m.pt` / custom `.pt`).
- Use Ultralytics exporter:
  - Set `imgsz=640`, **`batch=1` static**, `dynamic=False`.
  - Choose precision: FP32 / FP16 / INT8.
  - For INT8, pass dataset YAML so the exporter can feed calibration images (`fraction=1.0` for final builds).

### (B) Evaluate accuracy (mAP) on the **test** split
- Run `model.val(split="test")` for baseline `.pt` and each `.engine`.
- Report **mAP50**, **mAP50–95**, precision, recall.

### (C) Measure speed (two contexts we used)
- **Part 1 (video)**: end-to-end ms/frame & FPS on a 20-s clip.
- **Part 2 (dataset)**: images/sec (ms/image) on the **test** split, **batch=1**, report p50 & p95 latency.

> I keep the same `imgsz` and confidence threshold across models to make the comparison fair.

---

## 5) Interpreting results (what to expect)

- **Accuracy**
  - FP32/FP16 engines ≈ baseline `.pt`.
  - INT8: small mAP drop unless calibration is extensive and matches deployment.

- **Latency / Throughput**
  - FP16: consistent speedups (our results ~20–30% vs baseline).
  - INT8: usually the fastest and with the **best p95** (lowest tail latency).
  - If TRT FP32 doesn’t beat PyTorch: remember your timing includes pre/post steps (decode, letterbox, NMS, drawing) where TRT gives no advantage.

---

## 6) Common pitfalls (and how I avoided them)

- **Batch mismatch**  
  If you build INT8 with `batch=8` but infer with batch=1, you may hit shape assertions.  
  **Fix**: build with `batch=1` (static), or use `dynamic=True` and select a profile at runtime.

- **Wrong `imgsz`**  
  Export and inference should use the same `imgsz` for consistent accuracy and speed.

- **Weak calibration**  
  Too few / unrepresentative images → larger INT8 accuracy drop.  
  **Fix**: use hundreds of images from similar distribution; we used `fraction=1.0` on the dataset.

- **Non-persistent artifacts**  
  Colab `/content` is temporary. I copy engines/CSVs to Drive for persistence.

- **Unfair timing**  
  We warm up per model and keep pre/post identical across runs; we also report p50/p95 to expose jitter.

---

## 7) How this maps to our assignments

- **Part 1 (pretrained YOLO11m + video)**  
  - **Qualitative**: four 20-s videos (baseline vs TRT FP32/FP16/INT8).  
  - **Quantitative**: FP16 and INT8 had the biggest speedups; FP32 was similar/slightly slower due to equal pre/post.

- **Part 2 (custom YOLO11m + dataset)**  
  - **Accuracy parity**: FP32/FP16 matched baseline mAP on **test**.  
  - **INT8**: fastest throughput with a small mAP50–95 dip; best tail latency.


