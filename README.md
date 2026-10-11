# Disease Triage — Multimodal ML Pipeline

A multimodal machine-learning pipeline for automated disease triage, fusing **RGB images**, **thermal images**, and **tabular clinical features** to classify skin lesions and predict triage level.

## Project Structure

```
disease-triage/
├── data/                   # All data assets
│   ├── rgb/                # RGB dermoscopy images (HAM10000)
│   │   ├── raw/            # Raw downloaded data
│   │   └── processed/      # Processed splits
│   ├── thermal/            # Thermal imaging (Terravic Facial IR)
│   │   ├── raw/            # Raw downloaded data
│   │   └── processed/      # Processed splits
│   └── tabular/            # Structured clinical metadata (Disease Symptom Prediction)
│       ├── raw/
│       └── processed/
├── models/                 # Saved / exported model artifacts
│   ├── rgb_encoder/        # RGB image encoder weights
│   ├── thermal_encoder/    # Thermal encoder weights
│   ├── tabular_encoder/    # Tabular feature encoder weights
│   └── fusion/             # Fused multimodal model weights
├── src/                    # Source code
│   ├── data_loaders/       # Dataset & DataLoader implementations
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation & metric reporting
│   ├── models/             # Model architectures
│   └── utils/              # Shared utilities
├── checkpoints/            # Epoch & best-model checkpoints
├── configs/                # YAML configuration files
├── logs/                   # Training logs (CSV)
├── results/                # Evaluation results & plots
├── ui/                     # Streamlit demo app
├── notebooks/              # Jupyter exploration notebooks
├── requirements.txt        # Python dependencies
├── run_all.py              # Pipeline orchestrator
├── PROGRESS.md             # Progress log
└── README.md               # ← You are here
```

## Quick Start

### 1. Create & activate virtual environment
```bash
python3 -m venv venv && source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install additional dependencies
```bash
pip install kaggle huggingface_hub
```

### 4. Configure Kaggle API (for HAM10000 and tabular data)
```bash
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/ with:
# {"username": "your_username", "key": "your_api_key"}
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Run the full pipeline
```bash
# Run all phases
python run_all.py

# Or run specific phases
python run_all.py --phase 1  # Data acquisition only
python run_all.py --phase 2  # Fast baseline training
python run_all.py --phase 3  # Fusion model
python run_all.py --phase 4  # Full training
python run_all.py --phase 5  # UI
python run_all.py --phase 6  # Wrap-up

# Force re-run a phase
python run_all.py --phase 2 --force
```

### 6. Launch the UI
```bash
streamlit run ui/app.py
```

## Datasets

| Modality | Dataset | Source | Size | Classes |
|----------|---------|--------|------|---------|
| RGB | HAM10000 | Hugging Face (BMN95) | 10,015 images | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| Tabular | Disease Symptom Prediction | Kaggle (itachi9604) | 4,920 samples | 41 diseases |
| Thermal | Terravic Facial IR | OSU OTCBVS | ~21,000 images | 20 subjects (pretraining only) |

### Known Limitations

1. **Thermal Data Gap**: No publicly available medical fever-thermal dataset with paired temperature labels exists. The Terravic dataset contains thermal face images but no fever/vital sign annotations. We use fever symptoms from tabular data (`high_fever`, `mild_fever`, `chills`, `shivering`) as a proxy ground-truth signal. **Real paired thermal+fever data is needed for production use.**

2. **Class Imbalance**: HAM10000 has severe class imbalance (nv: 6,705 vs df: 115). We use balanced sampling and class weights.

3. **Domain Gap**: Terravic thermal faces (indoor/outdoor, glasses/hat variations) ≠ medical fever screening thermal images. The thermal encoder learns general thermal features but may not transfer perfectly.

4. **Tabular-Temporal Mismatch**: Symptom data is static; real triage would need temporal vitals trends.

## Model Architecture

### Modality Encoders
- **RGB Encoder**: EfficientNet-B0 (ImageNet pretrained) → 512-dim projection
- **Thermal Encoder**: EfficientNet-B0 (adapted for 1-channel) → 512-dim projection  
- **Tabular Encoder**: 3-layer MLP (173→512→256→256) with BatchNorm + Dropout

### Fusion
- Project all modalities to common dimension (512)
- Multi-head attention (4 heads) across modalities
- Fusion MLP (1536→512→512)

### Output Heads
1. **Fever Flag**: Binary classification (sigmoid)
2. **Visual Condition**: 7-class classification (HAM10000 classes)
3. **Triage Level**: 3-class (mild / moderate / refer)

## Training Configurations

### Fast Pass (Phase 2-3)
- Frozen backbones (RGB, Thermal)
- 20% data subset
- 8 epochs
- Batch size: 32 (RGB/Thermal), 64 (Tabular), 16 (Fusion)
- LR: 1e-3

### Full Training (Phase 4)
- Unfrozen backbones (lower LR: 1e-4)
- Full dataset
- 50-100 epochs with early stopping
- Cosine annealing LR schedule
- Class-weighted losses

## Evaluation Metrics

Results saved to `results/`:
- Per-class Precision, Recall, F1
- Macro & Weighted averages
- Confusion matrices
- Training curves

## Reproducing Results

```bash
# Fast pass (quick sanity check)
python run_all.py --phase 2 --force
python run_all.py --phase 3 --force

# Full training (run overnight)
python run_all.py --phase 4 --force

# Evaluate
python -m src.evaluation.evaluate_rgb --config configs/rgb_fast.yaml --checkpoint checkpoints/rgb/best_model.pt
python -m src.evaluation.evaluate_tabular --config configs/tabular_fast.yaml --checkpoint checkpoints/tabular/best_model.pt
python -m src.evaluation.evaluate_fusion --config configs/fusion_fast.yaml --checkpoint checkpoints/fusion/best_model.pt
```

## Expected Performance (Fast Pass Estimates)

| Model | Visual Acc | Fever AUC | Triage Acc |
|-------|-----------|-----------|------------|
| RGB Baseline | ~75% | N/A | N/A |
| Tabular Baseline | N/A | ~85% | ~70% |
| Fusion | ~78% | ~88% | ~75% |

*Full training should improve by 5-10% absolute.*

## License

This project is for research/educational purposes only. 
- HAM10000: CC BY-NC 4.0 (non-commercial)
- Terravic: IEEE OTCBWS Benchmark (research only)
- Disease Symptom Prediction: CC-BY-SA-4.0

## Citation

If you use this pipeline, please cite the original datasets:
- Tschandl et al., "The HAM10000 dataset", Scientific Data 2018
- Terravic Facial IR Database, OTCBVS Benchmark
- Disease Symptom Prediction dataset (Kaggle)

## Disclaimer

⚠️ **This is a decision-support demo, not a medical diagnosis tool.** The model has known limitations (especially thermal data proxy) and should never be used for clinical decision-making without rigorous validation and regulatory approval.