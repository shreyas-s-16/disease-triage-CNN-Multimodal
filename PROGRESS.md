# Disease Triage Pipeline — Progress Log

## Phase 1 — Data Acquisition

### Status: 🔄 In Progress

### RGB Branch (HAM10000)
- ✅ Downloaded via Hugging Face (BMN95/ham10000-skin-lesion-classifier-data) to `data/rgb/raw/`
- ✅ Balanced subset created: 1,757 images across 7 classes (300 each for nv, mel, bkl, bcc, akiec; 142 vasc; 115 df)
- ✅ Metadata saved: `data/rgb/raw/HAM10000_metadata_balanced_subset.csv`
- ⏳ **Full image download in progress** - Kaggle/Harvard Dataverse download running (267/10015 images so far)
- Class distribution in metadata: nv=6705, mel=1113, bkl=1099, bcc=514, akiec=327, vasc=142, df=115

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

## Phase 2 — Fix Fusion Design (Phase B)

### Status: ✅ Complete

### Design Decision: Unpaired Multimodal Training
- **Problem**: Current FusionDataset pairs i-th RGB, thermal, and tabular samples from unrelated datasets, so fever/triage labels have no relation to the image.
- **Solution**:
  1. Train visual head ONLY on real labelled HAM10000 images (RGB branch, frozen-then-unfrozen EfficientNet-B0)
  2. Train fever and triage heads ONLY on tabular symptom data
  3. Use thermal encoder as contrastively pre-trained features; do not fabricate paired labels
  4. Fusion at inference accepts any subset of modalities (missing-modality masking: zero the projection and mask in attention)
  4. During training, randomly drop modalities (modality dropout)

### Implementation:
- `ModalityDropoutDataset` in `src/training/train_fusion.py` with configurable dropout probability
- `FusionEncoder` with missing-modality masking via `key_padding_mask` in attention
- Modality dropout probability configurable in config (default 0.2)
- Updated `FusionModel.forward()` to handle any subset of modalities

---

## Phase 3 — Training Quality (Phase C)

### Status: ✅ Code Complete

### Implemented:
- Stratified 70/15/15 train/val/test splits for all modalities (saved to `data/*/processed/splits.csv`)
- Early stopping (patience 5, min_delta 1e-4) in all training scripts
- Resumable checkpoints (`checkpoint_last.pt` with optimizer/scheduler state)
- Cosine annealing LR scheduling
- Class-weighted losses
- Configs updated: `configs/*_full.yaml` with new parameters

---

## Phase 4 — Evaluation (Phase D)

### Status: ✅ Code Complete

### Implemented:
- `evaluate_rgb.py`: Generates metrics.json, confusion_matrix.png, training_curves.png, Grad-CAM overlays for 12 test images, sample_predictions.png grid
- `evaluate_tabular.py`: Generates metrics.json, confusion_matrix.png, training_curves.png with test splits
- `evaluate_fusion.py`: Generates metrics.json, confusion_matrix.png, training_curves.png with test splits
- All evaluations use stratified test splits and generate required artifacts in `results/`

---

## Phase 5 — UI (Phase E)

### Status: ✅ Code Complete

### Streamlit App (`ui/app.py`):
- Fixed symptom-count mismatch: loads symptom list from saved tabular encoder metadata (131 symptoms)
- Dynamic visual classes from config/checkpoint (7 HAM10000 classes)
- Proper missing-modality handling (zero tensors with masking in attention)
- Loads encoders from `checkpoints/tabular/` (disease_encoder.pkl, symptom_encoder.pkl)
- Medical disclaimer always visible
- Loads `best_model.pt` for fusion model

---

## Phase 6 — Delivery (Phase F)

### Status: 🔄 In Progress

### Remaining:
- Commit checkpoints/*/best_model.pt via Git LFS (or GitHub Release if >100 MB)
- Update README.md with "Design decision: unpaired multimodal training" section
- Update .gitignore to allow results/, screenshots/, logs/
- Push to GitHub and print commit URL

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

**Next Steps:**
1. Complete RGB image download (use Harvard Dataverse zip for bulk download)
2. Run Phase 4 full training with unfrozen backbones, more epochs
3. Evaluate on full validation sets
4. Launch UI for demo