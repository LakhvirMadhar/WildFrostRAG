"""Experiment config/results file I/O helpers.

Tracking (what experiments exist, their params/metrics, listing/searching)
is MLflow's job now (see services/evaluation/mlflow_tracking.py). This
package only holds the on-disk read/write helpers still used for
config.json/results.json/annotations.json - annotation is interactive
and hasn't moved to MLflow's artifact store yet.
"""
