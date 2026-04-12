# Machine Learning Playground

This repository is a collection of ML Projects, built primarily using in *Jupyter notebooks**. The project is collabaration between [Sasi Pawan](https://github.com/SasiPawan) and I. 


---

## Project 1: Regression Models Comparison

In this project, I built and compared multiple regression models within a unified pipeline to understand how different regularization techniques and optimization strategies affect performance. The goal was not just to implement models, but to systematically evaluate their behavior under consistent preprocessing and tuning conditions.

## What I Did:

Implemented a range of regression models including Linear Regression, Ridge, Lasso, Elastic Net, SGDRegressor, and Polynomial Regression to cover both basic and regularized approaches.

Designed a consistent preprocessing pipeline where categorical variables were transformed using one-hot encoding, ensuring that models could handle non-numeric data effectively.

Applied feature scaling using StandardScaler so that models sensitive to feature magnitudes (especially regularized and gradient-based ones) perform optimally.

Constructed evaluation functions to measure performance across multiple metrics, capturing both goodness-of-fit and error magnitudes.

Trained each model under the same data conditions to ensure a fair comparison across different algorithms.

Used GridSearchCV to systematically tune hyperparameters such as regularization strength, mixing ratios (for Elastic Net), and polynomial degrees.

## Key Observations:

Linear Regression performed well as a baseline but showed susceptibility to overfitting in the presence of multicollinearity and higher-dimensional feature spaces.

Ridge Regression helped stabilize the model by shrinking coefficients, improving generalization especially when features were highly correlated.

Lasso Regression introduced sparsity by driving some coefficients to zero, effectively performing implicit feature selection.

Elastic Net balanced both L1 and L2 penalties, often performing better when neither pure Ridge nor pure Lasso was ideal.

SGDRegressor demonstrated the importance of optimization strategy, particularly for larger datasets, though it required careful tuning of learning rates and regularization.

Polynomial Regression increased model flexibility but also significantly increased the risk of overfitting, especially at higher degrees.

## Takeaways:

Regularization plays a critical role in controlling model complexity and improving generalization, particularly in real-world datasets with noise and correlated features.

Different regression techniques are not strictly “better” or “worse”—their effectiveness depends heavily on the data distribution and feature relationships.

A unified pipeline with consistent preprocessing and evaluation is essential for making meaningful comparisons between models.

Hyperparameter tuning is not optional; it can drastically change model performance and often determines which model appears “best.”

---

## Project 2: K-Nearest Neighbors on PIMA Diabetes Dataset

In this mini-project, I implemented a K-Nearest Neighbors (KNN) classifier on the PIMA Diabetes dataset to predict whether a patient is diabetic or not. The primary goal was to build a robust preprocessing pipeline, handle missing/improper values intelligently, and evaluate how well a distance-based model performs on medical data.

## What I Did:

Loaded the dataset using KaggleHub and performed an initial inspection to understand feature distributions and anomalies.

Identified invalid zero values in medical attributes such as Glucose, Blood Pressure, BMI, Skin Thickness, and Insulin, and handled them using appropriate imputation strategies.

Applied median imputation for features with fewer zero values (Glucose, BloodPressure, BMI) to preserve statistical robustness.

Used a KNN-based imputation pipeline (with scaling) for features with a large number of missing values (SkinThickness, Insulin), allowing values to be inferred based on similarity across samples.

Split the dataset into train, validation, and test sets to ensure proper evaluation and avoid data leakage.

Built a pipeline combining StandardScaler and KNeighborsClassifier to ensure distance computations are meaningful across features with different scales.

Trained the model with initial hyperparameters (k = 9, Euclidean distance, uniform weights) and evaluated it on validation data.

Performed hyperparameter tuning using GridSearchCV over different values of k, distance metrics, and weighting strategies.

## Key Observations:

The dataset contains a significant number of zero values in features where zeros are not physiologically valid, making preprocessing a critical step for meaningful model performance.

KNN imputation worked well for highly sparse features like Insulin and SkinThickness, as it leverages patterns from similar samples rather than relying on global statistics.

Feature scaling had a strong impact on performance since KNN is distance-based and sensitive to magnitude differences across features.

The model achieved moderate performance, with an accuracy of around 57% on evaluation, but showed a strong bias toward predicting the majority class (non-diabetic).

The recall for diabetic patients was particularly low (~18%), indicating that the model struggles to correctly identify positive cases — a critical issue in medical diagnosis.

Hyperparameter tuning showed the best configuration to be k = 9, Euclidean distance, and uniform weighting, achieving a cross-validation accuracy of ~76%, though this did not fully translate to strong generalization.

## Takeaways:

This project highlights that preprocessing and data quality often matter more than model complexity, especially for simple algorithms like KNN.

While KNN is intuitive and easy to implement, it may not be the best choice for imbalanced medical datasets without additional techniques such as resampling or class weighting.

Evaluation metrics beyond accuracy (like recall and F1-score) are essential in healthcare applications, where false negatives can have serious consequences.

---

## Project 3: Dimensionality Reduction: GloVe Embeddings
In this mini-project, I explored GloVe (Global Vectors for Word Representation) using the glove.6B.200d pre-trained embeddings. My objective was to analyze how semantic relationships between words are preserved when visualized using dimensionality reduction techniques.

## What I Did:
Loaded 200-dimensional GloVe vectors and extracted embeddings for a carefully chosen set of overlapping words (e.g., apple, mango, computer).

Applied PCA followed by t-SNE to reduce dimensions to 2D for visualization.

Plotted word vectors using different perplexity values (25, 30, 40, 45, 50) to observe how clustering behavior changes.

## Key Observations:
At perplexity 25 and 30, the visualizations grouped semantically related words together meaningfully.
Notably, the word apple appeared between mango and computer, effectively capturing its dual nature as both a fruit and a tech brand.

At perplexity 40, the word apple shifted far from mango, indicating a loss in that semantic balance.

---