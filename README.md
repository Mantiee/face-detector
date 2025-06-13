# Face Detector Automation System

This project is a facial recognition system designed to trigger actions based on the presence or absence of a specific face. It uses [InsightFace](https://github.com/deepinsight/insightface) for accurate, GPU-accelerated face recognition. This solution is made for personal use cases such as automatically locking the system, turning off the screen or unmounting VeraCrypt and hibernating your PC (therefor locking your drives) when your face is no longer detected.

---

## Features

- Fast face recognition using InsightFace (`buffalo_l` model)
- Real-time webcam detection with optional GUI
- Configurable via `.cfg` files
- Multiple operating modes:
  - Debug mode
  - Monitor sleep control
  - VeraCrypt unmount + hibernation
- Custom face recording tool
- Failsafe exit (`CTRL+M`)
- Works with GPU acceleration (if set up properly)

---

## Requirements

- Python 3.8+
- Windows 10/11
- NVIDIA GPU (optional but recommended for real-time performance)
- VeraCrypt (optional, only needed for VeraCrypt mode)

---

## Installation

1. **Clone the repository** or download the files.

2. **Install dependencies:**

   You can use the `.bat` installer provided:
   ```
   install_dependencies.bat
   ```

   Or manually install:
   ```bash
   pip install opencv-python insightface numpy keyboard
   ```

3. **Download ONNX model weights:**

   InsightFace will automatically download the required ONNX models (`buffalo_l`) on first run. If needed, you can also download them manually from the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo).

---

## GPU Acceleration

This project is GPU-enabled via ONNX Runtime. To ensure GPU support:

1. Install the correct version of `onnxruntime-gpu`:

   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

2. Make sure you have the latest NVIDIA drivers and CUDA installed. CUDA 11.x or 12.x is typically required, depending on your GPU.

3. The face analysis module is initialized with:

   ```python
   FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
   ```

   You can verify GPU usage with a monitoring tool like `nvidia-smi`.

---

### cuDNN requirement for GPU support

To use GPU acceleration, ONNX Runtime requires access to the cuDNN libraries. Specifically, the `onnxruntime-gpu` package looks for DLL files such as `cudnn64_*.dll`.

You **must**:

1. Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) that matches your installed CUDA version (e.g. CUDA 12.2 → cuDNN 9.x).
2. Extract the downloaded files.
3. Ensure that the directory containing `cudnn64_*.dll` is either:
   - Added to your system `PATH`, or
   - Placed in the same directory as your Python interpreter or the `onnxruntime_providers_cuda.dll`.

If `cudnn64_9.dll` is missing, you will see an error like:
```
error loading onnxruntime_providers_cuda.dll which depends on "cudnn64_9.dll" which is missing.
```
Make sure to verify compatibility between your CUDA, cuDNN, and `onnxruntime-gpu` versions.

## Usage

### 1. Record your face

Run the face recording utility to collect reference embeddings:

```bash
record_reference.bat
```

This launches the camera, records for the configured number of seconds, and saves your face embeddings in `reference_embeddings.pkl`.

### 2. Start detection

After creating your reference profile, start the detection script:

```bash
run_detector.bat
```

The script loads the webcam and checks for your face. If not seen for a configurable timeout, it performs an action (monitor off, hibernate, etc.) depending on mode.

### 3. Configuration

Both scripts use `.cfg` files to define runtime parameters (e.g., camera index, duration, thresholds, GUI mode). You can modify:

- `main_config.cfg` — used by the main face detector
- `recorder_config.cfg` — used by the face recorder

---

## Notes

- VeraCrypt must be available in your system `PATH` if you want to use VeraCrypt mode.
- Use `CTRL+M` at any time to forcefully terminate the detector script.
- Monitor on/off is managed using PowerShell via WinAPI (no third-party tools needed).

---

## License

MIT License
