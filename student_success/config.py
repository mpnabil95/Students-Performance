"""Frozen experiment protocol. Change this before, never after, final evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'raw' / 'students.csv'
ARTIFACT_DIR = ROOT / 'artifacts'
REPORT_DIR = ROOT / 'reports'
SEED = 42
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']
PROTOCOL = {
    'version': 'semester1-v1',
    'scenario': 'Retrospective status prediction using information through semester 1',
    'target': 'Status at the end of the normal course duration; not time-to-dropout',
    'seed': SEED,
    'split': '60% model development / 20% policy validation / 20% historical holdout',
    'holdout_caveat': 'Same historical 20% split as the submission; already inspected in the old project, not pristine external validation.',
    'selection': 'Highest mean 5-fold macro F1; candidate order breaks exact ties.',
    'calibration': '3-fold sigmoid calibration within each training fold, ensemble=True',
    'policy': 'One review threshold maximizing dropout F2 on policy validation, tie -> higher threshold.',
    'threshold_grid': {'min': 0.05, 'max': 0.95, 'step': 0.01},
    'deployment_fit': 'Selected calibrated model fit only on 60% development; no post-holdout refit.',
    'feature_policy': '14 admission/semester-1 fields; exclude semester 2, ambiguously timed financial/macroeconomic fields, gender/nationality and family attributes. Age at enrollment is retained and audited.',
}
