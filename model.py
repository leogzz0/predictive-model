import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# function to load and clean data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.dropna(subset=['student.term_gpa_program', 'student_grades.final_numeric_afterAdjustment'])

    # Encode subcompetence levels
    level_encoder = LabelEncoder()
    df['subcompetence_level_encoded'] = level_encoder.fit_transform(
        df['subcompetence.level_assigned'].fillna('Not Assigned')
    )

    return df

# prepare data for model
def prepare_data(df):
    features = df[['student.term_gpa_program', 'subcompetence_level_encoded']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(features)

    y = df['student_grades.final_numeric_afterAdjustment'].apply(
        lambda x: 3 if x >= 90 else (2 if x >= 80 else (1 if x >= 70 else 0))
    )
    return X_poly, y

# generate simulated data for experiments
def generate_simulated_data(num_samples, gpa_range, subcompetence_levels):
    return pd.DataFrame({
        'student.term_gpa_program': np.random.uniform(gpa_range[0], gpa_range[1], num_samples),
        'subcompetence_level_encoded': np.random.choice(subcompetence_levels, num_samples)
    })

# simulate predictions
def simulate(model, new_data):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    new_data_poly = poly.fit_transform(new_data)
    predictions = model.predict(new_data_poly)
    probabilities = model.predict_proba(new_data_poly)
    return predictions, probabilities

# plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()

# plot probabilities
def plot_prediction_probabilities(probabilities, categories):
    for i, probs in enumerate(probabilities):
        plt.bar(categories, probs)
        plt.title(f"Prediction Probabilities for Sample {i+1}")
        plt.xlabel("Performance Category")
        plt.ylabel("Probability")
        plt.show()

# plot ROC curves
def plot_roc_curves(y_test_bin, y_scores, categories):
    for i in range(len(categories)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, label=f'Class {categories[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve by Class')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()


def main():
    file_path = 'EMCS-MC.csv'
    df = load_and_clean_data(file_path)

    # Prepare data
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)

    # Experiment 1: Low GPA
    simulated_data_low_gpa = generate_simulated_data(10, (50, 70), [0, 1, 2])
    print("\nSimulated Predictions for Low GPA:")
    low_gpa_predictions, low_gpa_probabilities = simulate(
        model, simulated_data_low_gpa)
    print(low_gpa_predictions)

    # Experiment 2: Average GPA with varied subcompetence
    simulated_data_avg_gpa = generate_simulated_data(10, (70, 80), [0, 1, 2])
    print("\nSimulated Predictions for Average GPA:")
    avg_gpa_predictions, avg_gpa_probabilities = simulate(
        model, simulated_data_avg_gpa)
    print(avg_gpa_predictions)

    # Experiment 3: High GPA
    simulated_data_high_gpa = generate_simulated_data(10, (90, 100), [0, 1, 2])
    print("\nSimulated Predictions for High GPA:")
    high_gpa_predictions, high_gpa_probabilities = simulate(
        model, simulated_data_high_gpa)
    print(high_gpa_predictions)

    # Evaluate model with test data
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, labels=['BAJO', 'PROMEDIO', 'BUENO', 'EXCELENTE'])

    # Plot probabilities
    categories = ['BAJO', 'PROMEDIO', 'BUENO', 'EXCELENTE']
    plot_prediction_probabilities(low_gpa_probabilities, categories)
    plot_prediction_probabilities(avg_gpa_probabilities, categories)
    plot_prediction_probabilities(high_gpa_probabilities, categories)

    # Plot ROC curves
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_scores = model.predict_proba(X_test)
    plot_roc_curves(y_test_bin, y_scores, categories)


if __name__ == "__main__":
    main()
