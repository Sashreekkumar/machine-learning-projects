# Machine Learning Playground

A collection of hands-on machine learning projects focused on **model comparison, preprocessing, and real-world constraints**.  
Built primarily using Jupyter notebooks.

---

## Index

- [Project 1: Regression Models Comparison](#project-1-regression-models-comparison)
- [Project 2: KNN on PIMA Diabetes Dataset](#project-2-knn-on-pima-diabetes-dataset)
- [Project 3: GloVe Embeddings Visualization](#project-3-glove-embeddings-visualization)
- [Project 4: XGBoost on Malaria Genomics](#project-4-xgboost-on-malaria-genomics)
- [Tech Stack](#tech-stack)
- [Notes](#notes)

---

## Project 1: Regression Models Comparison

Comparison of multiple regression techniques under a unified pipeline to study the impact of **regularization and optimization**.

**Models:** Linear, Ridge, Lasso, Elastic Net, SGDRegressor, Polynomial  

**Key Work:**
- One-hot encoding + feature scaling (StandardScaler)
- Unified training + evaluation pipeline
- GridSearchCV for hyperparameter tuning

**Insights:**
- Ridge stabilizes multicollinearity; Lasso performs feature selection  
- Elastic Net balances both worlds effectively  
- Polynomial models overfit quickly without regularization 

**Code:**
[Regression_Pipeline_ipynb](Regression/Regression_Pipeline_ipynb.ipynb)

---

## Project 2: KNN on PIMA Diabetes Dataset

Built a KNN classifier with a focus on **data cleaning and imputation** for medical data.

**Key Work:**
- Handled invalid zero values using median + KNN imputation  
- Scaled features for distance-based learning  
- Train/val/test split with pipeline-based modeling  

**Results:**
- Accuracy: ~57%  
- Strong class imbalance → poor recall for diabetic cases (~18%)

**Insights:**
- Preprocessing > model choice for KNN  
- Distance-based models struggle on imbalanced datasets  

**Code:**
[KNN.ipynb](KNN/KNN.ipynb)

---

## Project 3: GloVe Embeddings Visualization

Explored semantic relationships in **GloVe word embeddings** using dimensionality reduction.

**Key Work:**
- Used `glove.6B.200d` embeddings  
- PCA → t-SNE (2D visualization)  
- Tested multiple perplexity values (25–50)

**Insights:**
- Semantic clustering visible at lower perplexities  
- “apple” positioned between fruit + tech contexts  
- Higher perplexity distorted semantic relationships  

**Code:**
[glove-embeddings.ipynb](Dimensionality_Reduction/glove-embeddings.ipynb)

---

## Project 4: XGBoost on Malaria Genomics

Baseline classification model for **species prediction using SNP data (~48.5M features/sample)**.

**Key Work:**
- MAF filtering to remove uninformative variants  
- 0/1/2 SNP encoding  
- Sparse processing + `np.memmap` for memory efficiency  
- TruncatedSVD for dimensionality reduction  
- XGBoost classifier

**Results:**
- Accuracy: ~80%  
- Macro F1: ~0.60  

**Insights:**
- Dimensionality reduction is critical in genomics  
- XGBoost handles high-dimensional tabular data well  
- Performance limited by class imbalance + missing classes  

**Code Repository:**
[XG Boost on Malaria Genomics](https://github.com/Sashreekkumar/malaria-gen/blob/main/README.md)

---

## Tech Stack

`scikit-learn` • `numpy` • `pandas` • `matplotlib` • `XGBoost` • `PyTorch`

---

## Notes

This repository focuses on **learning through implementation**, with emphasis on:
- clean pipelines  
- fair model comparison  
- practical constraints (memory, data quality, imbalance)  