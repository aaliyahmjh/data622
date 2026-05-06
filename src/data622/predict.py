import numpy as np
import pandas as pd
import joblib
import os


def cast_to_string(x):
    return x.astype(str)

class NYCSalaryStabilizer:
    def __init__(self, model_path="models/salary_model_best.pkl", 
                 ref_table_path="data/processed/reference_table.csv"):
        if not os.path.exists(model_path):
            print(f"⚠️ Champion model not found at {model_path}.")
            # Backup file is the linear model
            if os.path.exists("models/salary_model_linear.pkl"):
                print("Falling back to linear model for now...")
                model_path = "models/salary_model_linear.pkl"
            else:
                raise FileNotFoundError("❌ No models found!")
            
        print(f"Loading model: {model_path}")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.feature_cols = bundle["feature_cols"]
        
        print(f"Loading reference data: {ref_table_path}")
        self.ref_table = pd.read_csv(ref_table_path)

    def predict_median_adjusted(self, user_input, model_weight=0.7):
        """
        Runs a prediction and adjusts it against the median salary for that title.
        This prevents the model from giving unrealistic results for 2024.
        """
        # Cap tenure at 20 years
        user_input['tenure_years'] = min(user_input.get('tenure_years', 0), 20)
        
        # Convert input to DataFrame and merge with median stats
        input_df = pd.DataFrame([user_input])
        input_df = input_df.merge(self.ref_table, on=['agency_std', 'title_std'], how='left')
        input_df = input_df.fillna(0)
        
        # Tenure Bucket
        input_df["tenure_bucket"] = pd.cut(
            input_df["tenure_years"],
            bins=[-np.inf, 0, 1, 3, 5, 10, 20, np.inf],
            labels=["0", "1", "2-3", "4-5", "6-10", "11-20", "20+"]
        ).astype(str)

        # Title Category 
        # User input doesn't match, default to 'Unknown'
        if 'title_category' not in input_df.columns:
            input_df['title_category'] = 'Unknown'

        # Group Rare Titles
        # Look at the historical count from the reference table
        job_count = input_df['count_of_job_titles'].values[0]
        input_df['title_std_grouped'] = input_df['title_std'] if job_count >= 100 else 'other_title'

        # Title Frequency 
        input_df['title_frequency'] = job_count

        # 5. Agency Size (This wasn't saved in the ref_table, so use a fallback average)
        if 'agency_size' not in input_df.columns:
            input_df['agency_size'] = 1000

        # Convert from Log to Dollars
        X_tf = self.preprocessor.transform(input_df[self.feature_cols])
        log_pred = self.model.predict(X_tf)[0]
        model_raw_dollars = np.exp(log_pred)
        
        # Median Adjustment Logic
        historical_median = input_df['median_salary_by_title'].values[0]
        
        if historical_median > 0:
            # Mix 70% model and 30% historical median
            final_salary = (model_raw_dollars * model_weight) + (historical_median * (1 - model_weight))
        else:
            # If it's a brand new title, we trust the model alone
            final_salary = model_raw_dollars

        return {
            "expected_salary": round(final_salary, 2),
            "historical_median": round(historical_median, 2),
            "model_prediction_raw": round(model_raw_dollars, 2)
        }

if __name__ == "__main__":
    # Internal Test Run
    try:
        stabilizer = NYCSalaryStabilizer()
        
        # Example
        sample_query = {
            'agency_std': 'police department', 
            'title_std': 'clerk', 
            'tenure_years': 5,
            'fiscal_year': 2024
        }
        
        result = stabilizer.predict_median_adjusted(sample_query)
        
        print("\n" + "="*40)
        print("NYC SALARY STABILIZER: TEST RESULT")
        print("="*40)
        print(f"Model Raw Guess:    ${result['model_prediction_raw']:,}")
        print(f"Historical Median:  ${result['historical_median']:,}")
        print(f"Adjusted Salary:    ${result['expected_salary']:,}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Could not run test: {e}")