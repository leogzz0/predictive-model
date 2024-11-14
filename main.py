import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# function to classify grades
def classify_grade(grade):
    if pd.isna(grade):
        return None
    if grade < 70:
        return 'BAJO'
    elif grade < 80:
        return 'PROMEDIO'
    elif grade < 90:
        return 'BUENO'
    else:
        return 'EXCELENTE'

# function to classify GPA
def classify_gpa(gpa):
    if pd.isna(gpa):
        return None
    if gpa < 70:
        return 0  # LOW
    elif gpa < 80:
        return 1  # AVERAGE
    elif gpa < 90:
        return 2  # GOOD
    else:
        return 3  # EXCELLENT

# load and clean data
def load_and_clean_data(file_path):
    # read data from csv file with low_memory=False to avoid dtype warnings
    df = pd.read_csv(file_path, low_memory=False)

    # Print initial data info
    print("\nInitial data shape:", df.shape)

    # eliminate rows with missing values
    df = df.dropna(subset=['student.term_gpa_program', 'student_grades.final_numeric_afterAdjustment'])

    print("Data shape after cleaning:", df.shape)

    # create new columns for grade and GPA categories
    df['grade_category'] = df['student_grades.final_numeric_afterAdjustment'].apply(
        classify_grade)
    df['gpa_category'] = df['student.term_gpa_program'].apply(classify_gpa)

    # convert subcompetence levels to numerical values
    level_encoder = LabelEncoder()
    df['subcompetence_level_encoded'] = level_encoder.fit_transform(
        df['subcompetence.level_assigned'].fillna('Not Assigned')
    )

    # Print value distributions
    print("\nGrade Category Distribution:")
    print(df['grade_category'].value_counts())

    print("\nGPA Category Distribution:")
    print(df['gpa_category'].value_counts())

    return df

# prepare data for analysis
def prepare_data_for_analysis(df):
    # create features for model
    features = df[['student.term_gpa_program', 'subcompetence_level_encoded']]

    # create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(features)

    # target variable: current grade category
    y = df['grade_category'].map({
        'BAJO': 0,
        'PROMEDIO': 1,
        'BUENO': 2,
        'EXCELENTE': 3
    })

    return X_poly, y

# subcompetence analysis
def analyze_competencies(df):
    # distribution analysis of subcompetence levels by GPA category
    competency_analysis = pd.crosstab(
        df['grade_category'],
        df['subcompetence.level_assigned'],
        normalize='index'
    )

    return competency_analysis

# visualization functions
def create_visualizations(df, competency_analysis):
    # configure the style of the visualizations
    plt.style.use('seaborn')

    # Create figure for grade distributions
    plt.figure(figsize=(12, 6))

    # Plot 1: Grade distribution
    plt.subplot(1, 2, 1)
    try:
        # Try using histplot first (newer seaborn versions)
        sns.histplot(
            data=df, x='student_grades.final_numeric_afterAdjustment', bins=20)
    except AttributeError:
        # Fall back to distplot for older seaborn versions
        sns.distplot(
            df['student_grades.final_numeric_afterAdjustment'], bins=20)

    plt.title('Grade Distribution')
    plt.xlabel('Grade')
    plt.ylabel('Frequency')

    # Plot 2: GPA vs Current Grades
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=df,
        x='student.term_gpa_program',
        y='student_grades.final_numeric_afterAdjustment',
        hue='grade_category'
    )
    plt.title('GPA vs Current Grades')
    plt.xlabel('GPA')
    plt.ylabel('Current Grade')

    plt.tight_layout()
    plt.show()

    # Subcompetence analysis visualization
    plt.figure(figsize=(10, 6))
    competency_analysis.plot(kind='bar', stacked=True)
    plt.title('Subcompetence Levels Distribution by Grade Category')
    plt.xlabel('Grade Category')
    plt.ylabel('Proportion')
    plt.legend(title='Subcompetence Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Additional visualization: Box plot of grades by competency level
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='subcompetence.level_assigned',
        y='student_grades.final_numeric_afterAdjustment'
    )
    plt.title('Grade Distribution by Subcompetence Level')
    plt.xlabel('Subcompetence Level')
    plt.ylabel('Grade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading and processing data...")
    df = load_and_clean_data('EMCS-MC.csv')

    print("\nPreparing data for analysis...")
    X_poly, y = prepare_data_for_analysis(df)

    # divide data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    # train logistic regression model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)

    # calculate model accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Testing accuracy: {test_accuracy:.2f}")

    # subcompetence analysis
    competency_analysis = analyze_competencies(df)
    print("\nCompetency Analysis:")
    print(competency_analysis)

    print("\nCreating visualizations...")
    # create visualizations
    create_visualizations(df, competency_analysis)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
