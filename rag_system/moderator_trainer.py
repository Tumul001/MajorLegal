"""
rag_system/moderator_trainer.py

Train a simple classifier to predict round winners (prosecution vs defense)
using features extracted from saved debate run logs in `data/debate_runs/`.

This provides a citation-aware calibration step: training the moderator to
replicate historical moderator judgments using structured features like
confidence scores, citation counts and provenance evidence strength.
"""
from typing import List, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def collect_run_files(runs_dir: str) -> List[Path]:
    p = Path(runs_dir)
    if not p.exists():
        return []
    return sorted([f for f in p.glob("*.json")])


def extract_features_from_run(run_file: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract per-round features and labels from one run file.

    Returns a DataFrame X and Series y.
    """
    with open(run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pros_args = data.get('prosecution_arguments', [])
    def_args = data.get('defense_arguments', [])
    verdicts = data.get('moderator_verdicts', [])

    rows = []
    labels = []

    n_rounds = min(len(pros_args), len(def_args), len(verdicts))
    for i in range(n_rounds):
        pa = pros_args[i]
        da = def_args[i]
        v = verdicts[i]

        # Features
        pros_conf = float(pa.get('confidence_score', 0) or 0)
        def_conf = float(da.get('confidence_score', 0) or 0)

        pros_cits = len(pa.get('case_citations', []) or [])
        def_cits = len(da.get('case_citations', []) or [])

        pros_reason_len = len((pa.get('legal_reasoning') or '').split())
        def_reason_len = len((da.get('legal_reasoning') or '').split())

        # Provenance quality: average evidence score if present
        def avg_prov_score(arg):
            prov = arg.get('provenance') or []
            scores = []
            for item in prov:
                for ev in item.get('evidence', []) or []:
                    s = ev.get('score')
                    if s is not None:
                        scores.append(float(s))
            return float(np.mean(scores)) if scores else 0.0

        pros_prov = avg_prov_score(pa)
        def_prov = avg_prov_score(da)

        row = {
            'pros_conf': pros_conf,
            'def_conf': def_conf,
            'pros_cits': pros_cits,
            'def_cits': def_cits,
            'pros_reason_len': pros_reason_len,
            'def_reason_len': def_reason_len,
            'pros_prov_score': pros_prov,
            'def_prov_score': def_prov,
            'conf_diff': pros_conf - def_conf,
            'cit_diff': pros_cits - def_cits,
            'prov_diff': pros_prov - def_prov
        }

        # Label: 1 if prosecution wins, 0 otherwise
        winner = v.get('round_winner', 'tie')
        label = 1 if winner == 'prosecution' else 0

        rows.append(row)
        labels.append(label)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y


def build_dataset(runs_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    files = collect_run_files(runs_dir)
    frames = []
    labels = []
    for f in files:
        X, y = extract_features_from_run(f)
        if X.empty:
            continue
        frames.append(X)
        labels.append(y)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int)

    X_all = pd.concat(frames, ignore_index=True)
    y_all = pd.concat(labels, ignore_index=True)
    return X_all, y_all


def train_and_save_model(runs_dir: str, out_model_path: str = 'data/models/moderator_model.joblib') -> dict:
    X, y = build_dataset(runs_dir)
    if X.empty:
        raise ValueError('No training data found in runs_dir')

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Save model
    out_path = Path(out_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)

    return {'accuracy': acc, 'report': report, 'model_path': str(out_path)}


def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', default='data/debate_runs')
    parser.add_argument('--out', default='data/models/moderator_model.joblib')
    args = parser.parse_args()

    res = train_and_save_model(args.runs_dir, args.out)
    print('Training complete:', res)
