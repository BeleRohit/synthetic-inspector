import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from ..schemas import Finding

def evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Finding:
    numeric_cols = real_df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        return Finding(
            name="Privacy Memorization",
            verdict="pass",
            score=1.0,
            details="No numeric columns to evaluate.",
            chart_html=""
        )
        
    X_real = real_df[numeric_cols].fillna(0)
    X_synth = synth_df[numeric_cols].fillna(0)
    
    scaler = MinMaxScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_synth_scaled = scaler.transform(X_synth)
    
    # Real-to-real distances (excluding self)
    nn_real = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_real.fit(X_real_scaled)
    distances_r2r, _ = nn_real.kneighbors(X_real_scaled)
    r2r_dist = distances_r2r[:, 1]  # index 0 is self
    
    threshold = np.percentile(r2r_dist, 5)
    
    # Synthetic-to-real distances
    nn_synth = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn_synth.fit(X_real_scaled)
    distances_s2r, _ = nn_synth.kneighbors(X_synth_scaled)
    s2r_dist = distances_s2r[:, 0]
    
    flagged_count = np.sum(s2r_dist < threshold)
    flagged_ratio = flagged_count / len(X_synth_scaled)
    
    if flagged_ratio > 0.01:
        verdict = "fail"
    elif flagged_ratio > 0.003:
        verdict = "warn"
    else:
        verdict = "pass"
        
    score = 1.0 - flagged_ratio
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s2r_dist, name='Synth-to-Real Distances', opacity=0.75
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                  annotation_text="5th %ile of Real-to-Real")
    fig.update_layout(title="Distance to Nearest Real Record", height=400)
    
    details = f"{flagged_count} out of {len(X_synth_scaled)} synthetic rows ({flagged_ratio*100:.2f}%) are closer to a real record than the 5th percentile threshold ({threshold:.3f})."
    
    return Finding(
        name="Privacy Memorization",
        verdict=verdict,
        score=score,
        details=details,
        chart_html=fig.to_html(full_html=False, include_plotlyjs='cdn')
    )
