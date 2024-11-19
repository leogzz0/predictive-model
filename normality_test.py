import pandas as pd
from scipy.stats import kstest, anderson
import matplotlib.pyplot as plt
import seaborn as sns

# function to load data and clean it
def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.dropna(subset=['student.term_gpa_program','student_grades.final_numeric_afterAdjustment'])
    return df

# function to run Kolmogorov-Smirnov test
def kolmogorov_smirnov_test(data, variable_name):
    stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f"Kolmogorov-Smirnov Test for {variable_name}:")
    print(f"  Statistic: {stat}")
    print(f"  p-value: {p_value}")
    if p_value < 0.05:
        print(
            f"  {variable_name} does NOT follow a normal distribution (reject H0)\n")
    else:
        print(
            f"  {variable_name} follows a normal distribution (fail to reject H0)\n")

# function to run Anderson-Darling test
def anderson_darling_test(data, variable_name):
    result = anderson(data, dist='norm')
    print(f"Anderson-Darling Test for {variable_name}:")
    print(f"  Statistic: {result.statistic}")
    print("  Critical Values and Significance Levels:")
    for i in range(len(result.critical_values)):
        sig_level = result.significance_level[i]
        crit_value = result.critical_values[i]
        print(f"    {sig_level}%: {crit_value}")
    # Typically use the 5% significance level
    if result.statistic > result.critical_values[2]:
        print(
            f"  {variable_name} does NOT follow a normal distribution (reject H0 at 5% level)\n")
    else:
        print(
            f"  {variable_name} follows a normal distribution (fail to reject H0 at 5% level)\n")

# function to visualize the distribution
def plot_distribution(data, variable_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"Distribution of {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("Frequency")
    plt.show()

def main():
    file_path = 'EMCS-MC.csv'
    df = load_data(file_path)

    # select variables
    gpa_data = df['student.term_gpa_program']
    grade_data = df['student_grades.final_numeric_afterAdjustment']

    # run tests for GPA
    print("=== Normality Tests for GPA ===")
    plot_distribution(gpa_data, "GPA")
    kolmogorov_smirnov_test(gpa_data, "GPA")
    anderson_darling_test(gpa_data, "GPA")

    # run tests for Grades
    print("\n=== Normality Tests for Final Grades ===")
    plot_distribution(grade_data, "Final Grades")
    kolmogorov_smirnov_test(grade_data, "Final Grades")
    anderson_darling_test(grade_data, "Final Grades")


if __name__ == "__main__":
    main()
