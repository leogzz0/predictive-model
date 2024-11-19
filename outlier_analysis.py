import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# function to load and clean data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['student.term_gpa_program', 'student_grades.final_numeric_afterAdjustment'])
    return df

# function to detect outliers using IQR method
def detect_outliers(data, variable_name):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    print(f"\nOutliers for {variable_name}:")
    print(outliers.describe())

    return outliers

# function to plot boxplot and highlight outliers
def plot_boxplot(data, variable_name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data, orient='h')
    plt.title(f"Boxplot for {variable_name}")
    plt.xlabel(variable_name)
    plt.show()


def main():
    file_path = 'EMCS-MC.csv'
    df = load_data(file_path)

    # analyze GPA
    print("=== GPA Outlier Analysis ===")
    gpa_data = df['student.term_gpa_program']
    gpa_outliers = detect_outliers(gpa_data, "GPA")
    plot_boxplot(gpa_data, "GPA")

    # analyze Final Grades
    print("\n=== Final Grades Outlier Analysis ===")
    grades_data = df['student_grades.final_numeric_afterAdjustment']
    grades_outliers = detect_outliers(grades_data, "Final Grades")
    plot_boxplot(grades_data, "Final Grades")


if __name__ == "__main__":
    main()
