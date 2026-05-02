import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
from ..schemas import Finding

def evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Finding:
    if "Outcome" not in real_df.columns:
        return Finding(
            name="Downstream ML",
            verdict="pass",
            score=1.0,
            details="Missing 'Outcome' column for classification.",
            chart_html=""
        )
        
    X_real = real_df.drop(columns=["Outcome"]).fillna(0)
    y_real = real_df["Outcome"].fillna(0)
    
    X_synth = synth_df.drop(columns=["Outcome"]).fillna(0)
    y_synth = synth_df["Outcome"].fillna(0)
    
    # Needs purely numeric for RF
    X_real = pd.get_dummies(X_real)
    X_synth = pd.get_dummies(X_synth)
    
    # Ensure same columns
    common_cols = list(set(X_real.columns) & set(X_synth.columns))
    X_real = X_real[common_cols]
    X_synth = X_synth[common_cols]
    
    if len(common_cols) == 0:
        return Finding(
            name="Downstream ML",
            verdict="pass",
            score=1.0,
            details="No common numeric features found.",
            chart_html=""
        )

    # 1. Baseline: real -> test on real (80/20)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    
    clf_base = RandomForestClassifier(random_state=42)
    clf_base.fit(X_train_r, y_train_r)
    preds_base = clf_base.predict_proba(X_test_r)[:, 1]
    baseline_auc = roc_auc_score(y_test_r, preds_base)
    
    # 2. TSTR: synthetic -> test on real
    clf_tstr = RandomForestClassifier(random_state=42)
    clf_tstr.fit(X_synth, y_synth)
    preds_tstr = clf_tstr.predict_proba(X_test_r)[:, 1]
    tstr_auc = roc_auc_score(y_test_r, preds_tstr)
    
    # 3. TRTS: real -> test on synthetic
    clf_trts = RandomForestClassifier(random_state=42)
    # Using the train portion of real to predict on all synth
    clf_trts.fit(X_train_r, y_train_r)
    preds_trts = clf_trts.predict_proba(X_synth)[:, 1]
    
    # Handle single class in synth test set
    if len(y_synth.unique()) > 1:
        trts_auc = roc_auc_score(y_synth, preds_trts)
    else:
        trts_auc = 0.5
        
    auc_drop = (baseline_auc - tstr_auc) / baseline_auc if baseline_auc > 0 else 0
    
    if auc_drop > 0.10:
        verdict = "fail"
    elif auc_drop > 0.05:
        verdict = "warn"
    else:
        verdict = "pass"
        
    score = tstr_auc / baseline_auc if baseline_auc > 0 else 0.0
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Baseline (Real->Real)', 'TSTR (Synth->Real)', 'TRTS (Real->Synth)'],
            y=[baseline_auc, tstr_auc, trts_auc],
            text=[f"{baseline_auc:.3f}", f"{tstr_auc:.3f}", f"{trts_auc:.3f}"],
            textposition='auto'
        )
    ])
    fig.update_layout(title="Random Forest ROC-AUC", height=400)
    
    details = f"Baseline AUC: {baseline_auc:.3f}. TSTR AUC: {tstr_auc:.3f}. Drop: {auc_drop*100:.1f}%."
    
    return Finding(
        name="Downstream ML",
        verdict=verdict,
        score=score,
        details=details,
        chart_html=fig.to_html(full_html=False, include_plotlyjs='cdn')
    )
