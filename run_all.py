#!/usr/bin/env python3
"""
Orchestrator script to run all phases of the disease-triage pipeline.
Can re-run any phase independently (skips phases with existing outputs).
"""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


PHASES = {
    1: {
        'name': 'Data Acquisition',
        'scripts': [],
        'outputs': [
            'data/rgb/raw/HAM10000_metadata_balanced_subset.csv',
            'data/rgb/raw/images',
            'data/tabular/raw/dataset.csv',
            'data/thermal/raw/terravic'
        ],
        'description': 'Download HAM10000, Disease Symptom Prediction, and Terravic datasets'
    },
    2: {
        'name': 'Fast Baseline Pass',
        'scripts': [
            ('RGB', 'src/training/train_rgb.py', 'configs/rgb_fast.yaml'),
            ('Tabular', 'src/training/train_tabular.py', 'configs/tabular_fast.yaml'),
            ('Thermal', 'src/training/train_thermal.py', 'configs/thermal_fast.yaml')
        ],
        'outputs': [
            'checkpoints/rgb/best_model.pt',
            'checkpoints/tabular/best_model.pt',
            'checkpoints/thermal/encoder_final.pt'
        ],
        'description': 'Train each modality encoder with fast settings (frozen backbone, 8 epochs, 20% data)'
    },
    3: {
        'name': 'Fusion Model',
        'scripts': [
            ('Fusion', 'src/training/train_fusion.py', 'configs/fusion_fast.yaml')
        ],
        'outputs': [
            'checkpoints/fusion/best_model.pt'
        ],
        'description': 'Train fusion model combining all three encoders'
    },
    4: {
        'name': 'Full Training Pass',
        'scripts': [
            ('RGB Full', 'src/training/train_rgb.py', 'configs/rgb_full.yaml'),
            ('Tabular Full', 'src/training/train_tabular.py', 'configs/tabular_full.yaml'),
            ('Thermal Full', 'src/training/train_thermal.py', 'configs/thermal_full.yaml'),
            ('Fusion Full', 'src/training/train_fusion.py', 'configs/fusion_full.yaml')
        ],
        'outputs': [
            'checkpoints/rgb/best_model.pt',
            'checkpoints/tabular/best_model.pt',
            'checkpoints/thermal/encoder_final.pt',
            'checkpoints/fusion/best_model.pt',
            'results/rgb/metrics.json',
            'results/tabular/metrics.json',
            'results/fusion/metrics.json'
        ],
        'description': 'Full training with unfrozen layers, more epochs, early stopping'
    },
    5: {
        'name': 'UI',
        'scripts': [],
        'outputs': ['ui/app.py'],
        'description': 'Streamlit app for inference demo'
    },
    6: {
        'name': 'Wrap-up',
        'scripts': [],
        'outputs': ['README.md', 'run_all.py'],
        'description': 'Update documentation and create orchestrator'
    }
}


def check_outputs_exist(outputs):
    """Check if all output paths exist."""
    for out in outputs:
        if not Path(out).exists():
            return False
    return True


def run_script(script_path, config_path):
    """Run a training script."""
    cmd = [sys.executable, script_path, '--config', config_path]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print(result.stdout)
    return True


def run_phase(phase_num, force=False):
    """Run a specific phase."""
    if phase_num not in PHASES:
        print(f"Unknown phase: {phase_num}")
        return False
    
    phase = PHASES[phase_num]
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num}: {phase['name']}")
    print(f"{'='*60}")
    print(phase['description'])
    
    # Check if outputs exist
    if not force and check_outputs_exist(phase['outputs']):
        print(f"✅ Phase {phase_num} outputs already exist. Skipping (use --force to re-run).")
        return True
    
    # Run scripts
    for name, script, config in phase['scripts']:
        print(f"\n--- {name} ---")
        if not Path(script).exists():
            print(f"Script not found: {script}")
            return False
        if not Path(config).exists():
            print(f"Config not found: {config}")
            return False
        
        if not run_script(script, config):
            print(f"Failed to run {name}")
            return False
    
    print(f"\n✅ Phase {phase_num} complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run disease-triage pipeline phases')
    parser.add_argument('--phase', type=int, choices=range(1, 7), help='Run specific phase')
    parser.add_argument('--force', action='store_true', help='Force re-run even if outputs exist')
    parser.add_argument('--start-from', type=int, default=1, help='Start from phase N')
    parser.add_argument('--list', action='store_true', help='List all phases')
    args = parser.parse_args()
    
    if args.list:
        print("Available phases:")
        for num, phase in PHASES.items():
            print(f"  {num}. {phase['name']}: {phase['description']}")
        return
    
    if args.phase:
        success = run_phase(args.phase, args.force)
        sys.exit(0 if success else 1)
    
    # Run all phases from start
    for phase_num in range(args.start_from, 7):
        success = run_phase(phase_num, args.force)
        if not success:
            print(f"\n❌ Phase {phase_num} failed. Stopping.")
            sys.exit(1)
    
    print("\n🎉 All phases completed successfully!")


if __name__ == '__main__':
    main()