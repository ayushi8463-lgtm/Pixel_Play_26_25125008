# 🔮 True Sight: Multi-Head Attention Anomaly Detection

**A Multi-Head Attention Autoencoder built for Pixel Play hackathon**

A deep learning approach for detecting anomalies in surveillance videos using a multi-head spatial attention autoencoder. This model has **True Sight** and it forces the AI to acknowledge the horror standing right in front of it!

## Features

- **Multi-Head Spatial Attention:** Captures different types of anomalies (large, medium, subtle) through parallel attention heads
- **Tight Bottleneck Architecture:** 32-channel bottleneck ensures poor reconstruction on anomalies while maintaining good reconstruction on normal frames
- **Multi-Scale Inference:** Tests at 5 different scales for robustness
- **Multi-Component Anomaly Scoring:** Combines 7 different error metrics for comprehensive anomaly detection
- **Temporal Smoothing:** Adaptive median filtering reduces false positives

## Performance
- Best Score: 0.658 (single run)
- Average: ~0.64 across multiple runs

## Getting Started: Your Exorcism Kit

### Prerequisites
```bash
pip install torch torchvision numpy pandas scipy pillow
```

### Required Dependencies
- Python 3.8+
- PyTorch 2.0+
- torchvision
- NumPy
- Pandas
- SciPy
- Pillow

### Dataset Structure: The Cursed Archives
```
Avenue_Corrupted/
├── Dataset/
│   ├── training_videos/          # The Cursed Archives
│   │   ├── 01/
│   │   │   ├── frame_00001.jpg
│   │   │   ├── frame_00002.jpg
│   │   │   └── ...
│   │   ├── 02/
│   │   └── ...
│   └── testing_videos/            # The Manifestation Tape
│       ├── 01/
│       ├── 02/
│       └── ...
```
### Usage: Performing the Exorcism

**Step 1: Configure the paths**
```python
TRAIN_DIR = 'path/to/training_videos'
TEST_DIR = 'path/to/testing_videos'
OUTPUT_PATH = 'submission.csv'
```

**Step 2: Run the exorcism**
```bash
python anomaly_detection.py
```

**Step 3: Review the prophecy**
```csv
Id,Predicted
1_1,0.234    # All clear
1_2,0.189    # All clear
1_3,0.847    # ⚠️ THE UNSEEN DETECTED
...
```

---

## Architecture: The True Sight Pipeline
```
Input Frame (3×96×96) 
    ↓
┌─────────────────────────────────┐
│   Encoder (Progressive Depth)   │
│   32 → 64 → 128 → 128 channels  │
│   High Dropout: 0.45-0.35       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Multi-Head Spatial Attention   │
│  ┌─────┐  ┌─────┐  ┌─────┐      │
│  │Head1│  │Head2│  │Head3│      │
│  └─────┘  └─────┘  └─────┘      │
│       Learnable Fusion          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│    Bottleneck (32 channels)     │
│    ← Critical Compression       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Decoder (Progressive Expand)  │
│   128 → 128 → 64 → 32 → 3       │
└─────────────────────────────────┘
    ↓
Reconstructed Frame (3×96×96)
```

### The Secret: Attention-Weighted Loss
```python
loss = (reconstruction_error * (1 + attention_map)).mean()
```
Forces the model to focus on regions it finds suspicious.

---
## Anomaly Scoring

The model uses a weighted combination of 7 error metrics to detect The Unseen:

| Metric | Weight | What It Detects |
|--------|--------|-----------------|
| **L1 Error** | 0.26 | Overall pixel-level disturbances |
| **L2 Error** | 0.22 | Squared differences (emphasizes large errors) |
| **Max Error** | 0.21 | Peak spatial anomalies |
| **Attention Error** | 0.15 | Regions the model finds suspicious |
| **Texture (Std) Error** | 0.08 | Unusual texture variations |
| **Frequency Error** | 0.06 | Abnormal frequency patterns |
| **Peak Signal Error** | 0.02 | Intensity spikes |

### Final Anomaly Score Formula
```python
anomaly_score = 0.26*L1 + 0.22*L2 + 0.21*Max + 0.15*Attention + 
                0.08*Std + 0.06*Freq + 0.02*Peak
```
___

## 🎓 Research Foundation:

This implementation draws from cutting-edge research:

### Primary Inspirations

**1. Kun Liu & Huadong Ma (2019)** - *"Exploring Background-bias for Anomaly Detection in Surveillance Videos"*
-  **Key Insight**: Models learn background patterns instead of anomalies
-  **My Solution**: Multi-head spatial attention explicitly focuses on anomalous regions

**2. Ren et al. (2021)** - *"Deep Video Anomaly Detection: Opportunities and Challenges"*
-  **Key Insight**: Hybrid approaches combining multiple techniques excel
-  **My Solution**: Multi-scale testing + multi-component scoring + temporal smoothing
___

### Research Directions
- Self-supervised pre-training on larger video datasets
- Few-shot learning for new anomaly types
- Explainable AI for interpretable anomaly reasoning

---

## License

This project was created for educational purposes as part of a hackathon.

---
