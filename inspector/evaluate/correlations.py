import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..schemas import Finding

def evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Finding:
    numeric_cols = real_df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        return Finding(
            name="Correlations",
            verdict="pass",
            score=1.0,
            details="Not enough numeric columns for correlation.",
            chart_html=""
        )

    real_corr = real_df[numeric_cols].corr()
    synth_corr = synth_df[numeric_cols].corr()
    
    diff_corr = np.abs(real_corr - synth_corr)
    mean_abs_diff = float(np.nanmean(diff_corr.values))
    max_abs_diff = float(np.nanmax(diff_corr.values))
    
    sign_flipped = False
    
    # Check for sign flips where |r| > 0.2 in real data
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            r_real = real_corr.iloc[i, j]
            r_synth = synth_corr.iloc[i, j]
            
            if pd.notna(r_real) and pd.notna(r_synth):
                if abs(r_real) > 0.2:
                    if r_real * r_synth < 0:
                        sign_flipped = True
                        break
        if sign_flipped:
            break
            
    if sign_flipped:
        verdict = "fail"
    elif max_abs_diff > 0.3:
        verdict = "warn"
    else:
        verdict = "pass"
        
    score = max(0.0, 1.0 - mean_abs_diff)
    
    # Plotting
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Real Correlation", "Synthetic Correlation"))
    
    fig.add_trace(go.Heatmap(
        z=real_corr.values,
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        showscale=False
    ), row=1, col=1)
    
    fig.add_trace(go.Heatmap(
        z=synth_corr.values,
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ), row=1, col=2)
    
    fig.update_layout(title_text="Correlation Matrices", height=500)
    
    details = f"Mean absolute difference: {mean_abs_diff:.3f}. Max absolute difference: {max_abs_diff:.3f}. Sign flips detected: {sign_flipped}."
    
    return Finding(
        name="Correlations",
        verdict=verdict,
        score=score,
        details=details,
        chart_html=fig.to_html(full_html=False, include_plotlyjs='cdn')
    )
