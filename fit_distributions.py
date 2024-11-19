import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# function to load and clean data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['student.term_gpa_program', 'student_grades.final_numeric_afterAdjustment'])
    return df

# function to fit distributions and calculate goodness of fit
def fit_distributions(data, variable_name):
    distributions = {
        'Exponencial': stats.expon,
        'Gamma': stats.gamma,
        'Weibull': stats.weibull_min,
        'Lognormal': stats.lognorm,
        'Pareto': stats.pareto
    }

    results = {}

    plt.figure(figsize=(12, 8))
    sns.histplot(data, kde=False, stat='density', bins=30, label='Data')

    for name, dist in distributions.items():
        params = dist.fit(data)
        pdf = dist.pdf(np.sort(data), *params)
        results[name] = {
            'params': params,
            'log_likelihood': np.sum(dist.logpdf(data, *params))
        }
        plt.plot(np.sort(data), pdf,label=f'{name} (LL: {results[name]["log_likelihood"]:.2f})')

    plt.title(f"Probability Distribution Fits for {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return results

# function to generate probability plot
def probability_plot(data, dist, params, variable_name):
    stats.probplot(data, dist=dist, sparams=params, plot=plt)
    plt.title(f'Probability Plot for {variable_name}')
    plt.show()

# function to generate QQ plot
def qq_plot(data, dist, params, variable_name):
    stats.probplot(data, dist=dist, sparams=params, plot=plt)
    plt.title(f'QQ Plot for {variable_name}')
    plt.show()


def main():
    file_path = 'EMCS-MC.csv'
    df = load_data(file_path)

    # fit distributions for GPA
    print("=== Fitting Distributions for GPA ===")
    gpa_data = df['student.term_gpa_program']
    gpa_results = fit_distributions(gpa_data, "GPA")

    # best fit distribution for GPA (Lognormal in this example)
    best_dist_gpa = stats.lognorm
    best_params_gpa = gpa_results['Lognormal']['params']
    probability_plot(gpa_data, best_dist_gpa, best_params_gpa, "GPA")
    qq_plot(gpa_data, best_dist_gpa, best_params_gpa, "GPA")

    # fit distributions for Final Grades
    print("\n=== Fitting Distributions for Final Grades ===")
    grades_data = df['student_grades.final_numeric_afterAdjustment']
    grades_results = fit_distributions(grades_data, "Final Grades")

    # best fit distribution for Final Grades (Lognormal in this example)
    best_dist_grades = stats.lognorm
    best_params_grades = grades_results['Lognormal']['params']
    probability_plot(grades_data, best_dist_grades, best_params_grades, "Final Grades")
    qq_plot(grades_data, best_dist_grades, best_params_grades, "Final Grades")


if __name__ == "__main__":
    main()
