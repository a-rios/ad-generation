import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', metavar='PATH',type=str, help="csv with durations, ADs, ratios")
    args = parser.parse_args()


    df =  pd.read_csv(args.csv)

    # Reshape the data for scikit-learn
    X = df[['duration']]  # Independent variable
    y = df['word_count']      # Dependent variable

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Retrieve the parameters
    intercept = model.intercept_
    slope = model.coef_[0]

    print(f"Linear Regression Equation: word_count = {intercept:.2f} + {slope:.2f} * duration_sec")
