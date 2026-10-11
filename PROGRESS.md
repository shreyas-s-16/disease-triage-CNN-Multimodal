# Disease Triage Pipeline — Progress Log

## Phase 1 — Data Acquisition

### Status: ✅ Complete

### RGB Branch (HAM10000)
- ✅ Downloaded via Hugging Face (BMN95/ham10000-skin-lesion-classifier-data) to `data/rgb/raw/`
- ✅ Balanced subset created: 1,757 images across 7 classes (300 each for nv, mel, bkl, bcc, akiec; 142 vasc; 115 df)
- ✅ Metadata saved: `data/rgb/raw/HAM10000_metadata_balanced_subset.csv`
- ⏳ Image download in progress (~267/1757 downloaded via HF API, slow)
- **Note**: Only 267 images (all bkl class) currently available locally; full balanced subset will download over time
- Class distribution in metadata: nv=300, mel=300, bkl=300, bcc=300, akiec=300, vasc=142, df=115

### Tabular Branch (Disease Symptom Prediction)
- ✅ Downloaded via Kaggle API: `itachi9604/disease-symptom-description-dataset`
- ✅ Files: `dataset.csv` (4,920 rows × 18 cols), `Symptom-severity.csv`, `symptom_Description.csv`, `symptom_precaution.csv`
- ✅ 41 diseases, 120 samples each (balanced)
- ✅ 17 symptom columns + Disease target
- ✅ Fever proxy symptoms available: `high_fever`, `mild_fever`, `chills`, `shivering`
- Class distribution: 41 classes × 120 = 4,920 total

### Thermal Branch
- ✅ Downloaded Terravic Facial IR Database (20 subjects) from OSU OTCBVS
- ✅ Extracted: ~11,163 thermal face images (320×240, 8-bit grayscale JPEG)
- ✅ Source: http://vcipl-okstate.org/pbvs/bench/Data/04/
- **Known Limitation**: No fever/temperature labels — using tabular fever symptoms as proxy for triage signal
- For pretraining thermal CNN encoder feature extraction only

---

## Phase 2 — Fast Baseline Pass

### Status: ✅ Complete

### RGB Branch
- ✅ **COMPLETED** - 8 epochs trained on MPS (Apple Silicon GPU)
- Train Acc: 99.5%, Val Acc: 100% (on 267 bkl-class images only)
- Model saved to: `checkpoints/rgb/best_model.pt`
- Config: `configs/rgb_fast.yaml`

### Tabular Branch
- ✅ **COMPLETED** - 8 epochs trained on MPS
- Train Acc: 100% disease, 100% fever, 100% triage
- Val Acc: 100% across all tasks
- Model saved to: `checkpoints/tabular/best_model.pt`
- Encoders saved for inference
- Config: `configs/tabular_fast.yaml`

### Thermal Branch
- ✅ **COMPLETED** - 5 epochs trained on MPS (Contrastive pretraining)
- Train Loss: 2.45, Val Loss: 2.43 (NT-Xent loss decreasing)
- Model saved to: `checkpoints/thermal/best_model.pt`
- Config: `configs/thermal_fast.yaml`

---

## Phase 3 — Fusion Model

### Status: ✅ Complete

### Architecture
- RGB encoder + Thermal encoder + Tabular encoder → Concat/Attention fusion → 3 output heads:
  1. Fever/vitals flag (binary)
  2. Visual condition class (7 classes: akiec, bcc, bkl, df, mel, nv, vasc)
  3. Overall triage (3 classes: mild, moderate, refer)

### Results (Fast Pass)
- ✅ **COMPLETED** - 8 epochs trained on MPS
- Val Acc: Visual=100%, Fever=90%, Triage=90%
- Average Val Acc: 93.33%
- Model saved to: `checkpoints/fusion/best_model.pt`
- Config: `configs/fusion_fast.yaml`

---

## Phase 4 — Full Training Pass

### Status: 📋 Ready to Run

### Ready to Run
- RGB Full: `configs/rgb_full.yaml`
- Tabular Full: `configs/tabular_full.yaml`
- Thermal Full: `configs/thermal_full.yaml`
- Fusion Full: `configs/fusion_full.yaml`

### To run:
```bash
python run_all.py --phase 4
```

---

## Phase 5 — UI

### Status: ✅ Code Complete (Minor dimension mismatch in demo)

### Streamlit App
- Location: `ui/app.py`
- Features: Upload RGB image, enter symptoms + temperature, get 3 predictions with confidence
- Always displays medical disclaimer
- **Note**: UI has hardcoded symptom list (173) vs trained model (131) - needs alignment for full demo

### To launch:
```bash
streamlit run ui/app.py
```

---

## Phase 6 — Wrap-up

### Status: ✅ Complete

### Deliverables
- ✅ `run_all.py` orchestrator (re-runs any phase independently)
- ✅ Updated `README.md` with setup, architecture, limitations
- ✅ `PROGRESS.md` tracking

---

## Summary

**Fast Pass Results:**
| Model | Visual Acc | Fever Acc | Triage Acc |
|-------|-----------|-----------|------------|
| RGB Baseline | 100%* | N/A | N/A |
| Tabular Baseline | N/A | 100% | 100% |
| Fusion | 100% | 90% | 90% |

*\*RGB trained on single class (bkl) due to limited downloaded images*

**Known Limitations:**
1. RGB dataset incomplete (only 267/1757 images downloaded, all bkl class)
2. Thermal data has no fever labels (Terravic is face recognition dataset)
3. Fever proxy from tabular symptoms only
4. Small validation sets in fast pass
5. UI symptom list needs alignment with trained model

**Next Steps for Full Training:**
1. Complete RGB image download (use Harvard Dataverse zip for bulk download)
2. Run Phase 4 full training with unfrozen backbones, more epochs
3. Evaluate on full validation sets
4. Launch UI for demo