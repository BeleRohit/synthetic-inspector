import pandas as pd
import numpy as np

from inspector.evaluate import marginals, correlations, subgroups, downstream, privacy

def main():
    print("Running Smoke Test...")
    
    # Create dummy Pima-like dataset (50 rows)
    np.random.seed(42)
    n = 50
    real_data = {
        "Pregnancies": np.random.randint(0, 10, n),
        "Glucose": np.random.randint(70, 180, n),
        "BloodPressure": np.random.randint(60, 100, n),
        "SkinThickness": np.random.randint(10, 40, n),
        "Insulin": np.random.randint(15, 200, n),
        "BMI": np.random.uniform(20.0, 40.0, n),
        "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.5, n),
        "Age": np.random.randint(21, 70, n),
        "Outcome": np.random.randint(0, 2, n)
    }
    
    real_df = pd.DataFrame(real_data)
    
    # Create a slightly perturbed synthetic dataset
    synth_data = real_data.copy()
    synth_data["Glucose"] = synth_data["Glucose"] + np.random.normal(0, 5, n)
    synth_data["Age"] = synth_data["Age"] + np.random.randint(-2, 3, n)
    
    synth_df = pd.DataFrame(synth_data)
    
    # Run Evaluators
    evaluators = [
        marginals.evaluate,
        correlations.evaluate,
        subgroups.evaluate,
        downstream.evaluate,
        privacy.evaluate
    ]
    
    print("\n--- Evaluation Results ---")
    for evaluator in evaluators:
        finding = evaluator(real_df, synth_df)
        print(f"[{finding.name}]")
        print(f"  Verdict: {finding.verdict}")
        print(f"  Score:   {finding.score:.3f}")
        print(f"  Details: {finding.details}\n")
        
    print("Smoke Test Completed Successfully!")

if __name__ == "__main__":
    main()
