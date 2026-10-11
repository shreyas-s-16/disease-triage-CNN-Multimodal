"""
Evaluation package for disease-triage pipeline.
"""
from .metrics import (
    evaluate_model,
    evaluate_multitask,
    plot_confusion_matrix,
    plot_training_history,
    save_evaluation_report,
    run_evaluation
)

__all__ = [
    'evaluate_model',
    'evaluate_multitask',
    'plot_confusion_matrix',
    'plot_training_history',
    'save_evaluation_report',
    'run_evaluation'
]