# YOLO11m → TensorRT (Assignments 1 & 2)

This repo contains two mini-projects:
- **Part 1**: Optimize pretrained YOLO11m with TensorRT (FP32/FP16/INT8), compare **video** inference speed vs baseline.
- **Part 2**: Optimize **custom-trained** YOLO11m, compare **test mAP** and **images/sec** vs baseline `.pt`.

## Environment (Colab)
- GPU: <Tesla T4 / A100>  
- Python: <3.10/3.11>  
- Torch: <2.x>  
- TensorRT: <10.x> via `pip --extra-index-url https://pypi.nvidia.com`  
- Ultralytics: <8.x>

### Part 1 — Video Inference Speed

| Model           | avg ms/frame | FPS  | timed frames |
|-----------------|--------------:|-----:|-------------:|
| PyTorch FP32    | 27.30         | 36.63 | 669         |
| TensorRT FP32   | 30.39         | 32.91 | 669         |
| TensorRT FP16   | 17.39         | 57.50 | 669         |
| TensorRT INT8   | 14.32         | 69.84 | 669         |

**Preview videos (Google Drive):**

**All videos (Drive folder):** <https://drive.google.com/drive/folders/1ecePVSIe8f9SeE78q3aq3kYRmuihbyhP?usp=drive_link> 

**Quantitative (20-s clip, 669 timed frames; warm-up 20 frames):**  

TensorRT delivered clear speedups over baseline PyTorch on video inference. FP16 ran at **17.39 ms/frame (57.50 FPS)**—about **36% lower latency** (+58% FPS) than baseline **27.30 ms (36.63 FPS)**. INT8 was fastest at **14.32 ms/frame (69.84 FPS)**—about **48% lower latency** (+91% FPS). TRT FP32 measured **30.39 ms (32.91 FPS)**, slightly slower than baseline, which is expected when end-to-end timing includes identical pre/post steps (decode, letterbox, NMS, drawing) that TensorRT doesn’t accelerate.


## Results (Part 2: custom dataset)
**Accuracy (test)**

| Model        | mAP50 | mAP50-95 | Precision | Recall |
|--------------|------:|---------:|----------:|-------:|
| Baseline .pt | 0.9171  | 0.3914     | 0.9690      | 0.8688   |
| TRT FP32     | 0.9178  | 0.3927     | 0.9669      | 0.8687   |
| TRT FP16     | 0.9178  | 0.3921     | 0.9668      | 0.8675   |
| TRT INT8     | 0.9174  | 0.3864     | 0.9703      | 0.8800   |

**Throughput (test images, batch=1)**

| Model        | avg ms/img | FPS | p50 ms | p95 ms |
|--------------|-----------:|----:|-------:|-------:|
| Baseline .pt | 26.65       | 37.52| 23.88   | 39.15   |
| TRT FP32     | 24.20       | 41.32| 23.43   | 29.44   |
| TRT FP16     | 20.87       | 47.92| 20.18   | 25.55   |
| TRT INT8     | 19.93       | 50.17| 18.50   | 24.51   |

**Key takeaways**
- FP16 ≈ baseline accuracy; ~20–30% lower latency.
- INT8 fastest; tiny mAP drop; best tail latency (p95).

_See `results/part2/*.csv` for raw tables of accuracy and throughput comparison of baseline and .engine models.

## Notes
- INT8 calibration used `val` split with fraction 1.0 and `imgsz=640`.
- Engines exported with `batch=1` (static) and `dynamic=False`.  
