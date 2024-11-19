# Student Academic Performance Prediction Model

## Overview

This repository contains a predictive model designed to analyze and classify student academic performance at TEC university. The model focuses on identifying key factors influencing student outcomes, particularly during their first semester, and leverages Machine Learning techniques to provide actionable insights.

## Objectives

The main objectives of this project are:
- **Classification of Student Performance**: Categorize students into performance levels (BAJO, PROMEDIO, BUENO, EXCELENTE) based on academic data.
- **Prediction of Academic Outcomes**: Predict final grades and GPA using logistic regression.
- **Scenario Simulation**: Explore hypothetical scenarios to understand the impact of variables such as GPA and subcompetence levels on student performance.
- **Data-Driven Decision Making**: Support academic interventions and resource allocation through data insights.

## Data Description

The dataset includes a comprehensive set of academic and demographic variables:
- **Student Demographics**: Age, gender, nationality.
- **Academic Background**: TEC vs non-TEC educational origin.
- **Course Information**: Subjects, grades, course types, and completion status.
- **Performance Metrics**: GPA, term performance, and final numeric grades.
- **Competency Assessments**: Transversal and disciplinary subcompetence levels.
- **Campus Information**: Regional campus and cohort details.
- **Academic Status Indicators**: Enrollment and academic standing.

### Key Features
- Student ID and cohort identification.
- Detailed course data including modalities and grades.
- Subcompetence levels (observed or not observed).
- Performance indicators such as GPA and competency achievement.

## Simulation Design

The simulation was designed to:
- **Predict Performance**: Use a multinomial logistic regression model to predict performance categories.
- **Evaluate Scenarios**: Analyze how changes in GPA and subcompetence levels affect predicted outcomes.
- **Assess Model Robustness**: Test the model under various experimental conditions (low, average, and high GPA scenarios).

### Simulation Experiments
1. **Low GPA Scenario**: Simulate students with GPA between 50 and 70.
2. **Average GPA Scenario**: Simulate students with GPA between 70 and 80.
3. **High GPA Scenario**: Simulate students with GPA above 90.

## Evaluation Metrics

The model’s performance was evaluated using the following metrics:
- **Confusion Matrix**: To measure classification accuracy and misclassification patterns.
- **ROC Curve and AUC**: To assess the model's ability to distinguish between performance categories.
- **Prediction Probabilities**: To visualize the confidence level of the model for each predicted class.

## Insights

Key insights from the analysis include:
- **High correlation between GPA and final grades**: Students with higher GPAs tend to perform better consistently.
- **Subcompetence levels as critical predictors**: Observed subcompetence levels strongly correlate with higher performance categories.
- **Performance in intermediate categories**: Higher uncertainty was observed in classifying PROMEDIO and BUENO categories, suggesting a need for more granular data or refined features.

## Proposed Improvements

Based on the analysis, the following improvements are suggested:
1. **Enriching the Dataset**: Include additional variables such as attendance, participation in extracurricular activities, and socioeconomic factors.
2. **Testing Alternative Models**: Explore models like Random Forests and Gradient Boosting for potentially higher accuracy.
3. **Preprocessing Enhancements**: Apply advanced techniques such as normalization and resampling to handle data imbalance.
4. **Scenario Analysis**: Continue simulating scenarios to explore how changes in student profiles impact predicted outcomes.

## Intended Use

This model provides a robust framework for:
- **Early Risk Identification**: Detect at-risk students and intervene proactively.
- **Resource Optimization**: Allocate academic support resources efficiently.
- **Policy Development**: Inform data-driven decisions to enhance student success and academic policies.
- **Academic Performance Insights**: Understand the role of subcompetence and GPA in shaping student outcomes.

## Visual Outputs

The key outputs include:
- **Confusion Matrix**: Evaluates classification performance.
- **ROC Curves**: Illustrates the model’s discriminative power.
- **Probability Distributions**: Displays prediction confidence for different scenarios.

## Conclusion

This project demonstrates the potential of Machine Learning in education to provide actionable insights and support institutional decision-making. Future iterations will aim to refine the model further and incorporate additional data to enhance predictive accuracy.
