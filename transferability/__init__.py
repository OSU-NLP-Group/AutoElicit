"""
Transferability Evaluation Module

This module evaluates how elicited unsafe behaviors transfer across
different computer-use agents (CUAs). It provides tools for:

- Executing agents on adversarial benchmark examples
- Analyzing trajectories for safety classification
- Computing transfer rates between source and target agents

See README.md for usage instructions.
"""

__all__ = [
    "run_benchmark",
    "run_benchmark_multiple_runs",
    "analyze_trajectories",
    "analyze_trajectories_multiple_runs",
    "summarize_results",
    "compute_transfer_rates",
]
