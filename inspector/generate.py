import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

def load_pima() -> pd.DataFrame:
    """
    Downloads the Pima dataset and adds column headers.
    Returns the DataFrame. Does NOT write to disk or GCS.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(url, names=columns)
    return df

def generate_synthetic(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses SDV GaussianCopulaSynthesizer to generate synthetic data.
    Fits on real_df, samples same number of rows.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_df)
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_df)
    
    synthetic_df = synthesizer.sample(num_rows=len(real_df))
    return synthetic_df
