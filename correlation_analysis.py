import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['student.term_gpa_program',
                   'student_grades.final_numeric_afterAdjustment'])
    return df

# Function to preprocess data: encode categorical variables


def preprocess_data(df):
    # Encode 'subcompetence.level_assigned' if it exists
    if 'subcompetence.level_assigned' in df.columns:
        df['subcompetence.level_assigned_encoded'] = df['subcompetence.level_assigned'].astype(
            'category').cat.codes
    return df

# Function to calculate and visualize correlation matrix


def correlation_matrix(df, variables_of_interest):
    # Filter data to include only numerical columns and variables of interest
    df_corr = df[variables_of_interest].corr()

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Correlation Matrix")
    plt.show()

    return df_corr

# Function to analyze significant correlations


def analyze_correlations(corr_matrix, threshold=0.5):
    # Identify correlations above threshold (positive or negative)
    high_corr = corr_matrix[(corr_matrix.abs() > threshold)
                            & (corr_matrix.abs() < 1.0)]

    print("\nSignificant Correlations (|r| > 0.5):")
    for row, col in high_corr.stack().index:
        if row != col:  # Avoid self-correlation
            print(f"{row} vs {col}: Correlation = {high_corr[row][col]:.2f}")
    return high_corr


def main():
    file_path = 'EMCS-MC.csv'
    df = load_data(file_path)

    # Preprocess data
    df = preprocess_data(df)

    # Define variables of interest
    variables_of_interest = [
        'student.term_gpa_program',
        'student_grades.final_numeric_afterAdjustment',
        'subcompetence.level_assigned_encoded'
    ]

    print("\n=== Correlation Matrix ===")
    corr_matrix = correlation_matrix(df, variables_of_interest)

    print("\n=== Analysis of Correlations ===")
    analyze_correlations(corr_matrix, threshold=0.5)


if __name__ == "__main__":
    main()
