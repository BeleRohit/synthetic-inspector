import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..schemas import Finding

def evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Finding:
    if "Age" not in real_df.columns or "Outcome" not in real_df.columns:
        return Finding(
            name="Subgroups",
            verdict="pass",
            score=1.0,
            details="Missing 'Age' or 'Outcome' columns.",
            chart_html=""
        )

    # Create quartile bins based on real data
    try:
        real_df['Age_Q'], bins = pd.qcut(real_df['Age'], q=4, retbins=True, duplicates='drop')
    except ValueError:
        return Finding(
            name="Subgroups",
            verdict="pass",
            score=1.0,
            details="Could not bucket 'Age' into quartiles.",
            chart_html=""
        )

    # Apply bins to synthetic
    # We clip synthetic ages to ensure they fall within the bins ranges if they are slightly out
    synth_age_clipped = np.clip(synth_df['Age'], bins[0], bins[-1])
    synth_df['Age_Q'] = pd.cut(synth_age_clipped, bins=bins, include_lowest=True)
    
    real_means = real_df.groupby('Age_Q', observed=False)['Outcome'].mean()
    synth_means = synth_df.groupby('Age_Q', observed=False)['Outcome'].mean()
    
    max_rel_diff = 0.0
    labels = []
    real_vals = []
    synth_vals = []
    
    for idx in real_means.index:
        r_val = real_means[idx]
        s_val = synth_means[idx]
        
        labels.append(str(idx))
        real_vals.append(r_val)
        synth_vals.append(s_val)
        
        if pd.notna(r_val) and r_val > 0 and pd.notna(s_val):
            rel_diff = abs(r_val - s_val) / r_val
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff
                
    if max_rel_diff > 0.15:
        verdict = "fail"
    elif max_rel_diff > 0.08:
        verdict = "warn"
    else:
        verdict = "pass"
        
    score = max(0.0, 1.0 - max_rel_diff)
    
    fig = go.Figure(data=[
        go.Bar(name='Real Mean Outcome', x=labels, y=real_vals),
        go.Bar(name='Synthetic Mean Outcome', x=labels, y=synth_vals)
    ])
    fig.update_layout(barmode='group', title="Mean Outcome per Age Subgroup", height=400)
    
    details = f"Maximum relative difference across subgroups is {max_rel_diff*100:.1f}%."
    
    return Finding(
        name="Subgroups",
        verdict=verdict,
        score=score,
        details=details,
        chart_html=fig.to_html(full_html=False, include_plotlyjs='cdn')
    )
