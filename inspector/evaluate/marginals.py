import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from ..schemas import Finding

def evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Finding:
    numeric_cols = real_df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        return Finding(
            name="Marginal Distributions",
            verdict="pass",
            score=1.0,
            details="No numeric columns to evaluate.",
            chart_html=""
        )
        
    p_values = []
    
    # Calculate grid size for subplots
    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)
    
    for i, col in enumerate(numeric_cols):
        # KS test
        stat, p_val = stats.ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
        p_values.append(p_val)
        
        # Plotting
        row = (i // n_cols) + 1
        col_idx = (i % n_cols) + 1
        
        fig.add_trace(go.Histogram(
            x=real_df[col], name='Real', marker_color='blue', opacity=0.5,
            showlegend=(i==0)
        ), row=row, col=col_idx)
        
        fig.add_trace(go.Histogram(
            x=synth_df[col], name='Synthetic', marker_color='coral', opacity=0.5,
            showlegend=(i==0)
        ), row=row, col=col_idx)

    fig.update_layout(barmode='overlay', title_text="Marginal Distributions", height=300*n_rows)
    
    # Logic for verdict
    failed_cols = sum(1 for p in p_values if p < 0.05)
    fail_ratio = failed_cols / len(numeric_cols)
    
    if fail_ratio > 0.30:
        verdict = "fail"
    elif fail_ratio > 0.15:
        verdict = "warn"
    else:
        verdict = "pass"
        
    score = 1.0 - fail_ratio
    
    details = f"{failed_cols} out of {len(numeric_cols)} columns ({fail_ratio*100:.1f}%) failed the KS test (p < 0.05)."
    
    return Finding(
        name="Marginal Distributions",
        verdict=verdict,
        score=score,
        details=details,
        chart_html=fig.to_html(full_html=False, include_plotlyjs='cdn')
    )
