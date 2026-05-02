from typing import Literal
from pydantic import BaseModel

class Finding(BaseModel):
    name: str
    verdict: Literal["pass", "warn", "fail"]
    score: float  # 0.0 to 1.0, higher = better fidelity
    details: str
    chart_html: str  # Plotly fig.to_html(full_html=False)
