import os
from jinja2 import Environment, FileSystemLoader
from .schemas import Finding

VERDICT_RANK = {"pass": 0, "warn": 1, "fail": 2}

def assemble(findings: list[Finding]) -> str:
    """
    Compute overall verdict and render the HTML report.
    """
    if not findings:
        overall_verdict = "pass"
    else:
        overall = max(findings, key=lambda f: VERDICT_RANK[f.verdict])
        overall_verdict = overall.verdict

    # Set up Jinja2 environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(os.path.dirname(current_dir), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("report.html")
    
    # Render HTML
    html_out = template.render(
        findings=findings,
        overall_verdict=overall_verdict
    )
    
    return html_out
