# Data Science Internship Project: Statistical Inference, Machine Learning and Strategic Reporting

Yuva Intern, Weeks 3 to 5. An end-to-end data science project made of two case studies, each documented in a written report, and a capstone report that turns the results into strategic recommendations.

> **Disclaimer:** The e-commerce dataset is self-generated (simulated), and the tumour model is a learning project on a public benchmark dataset. Business-impact figures are illustrative, and the model must not be used for real medical diagnosis.

## Case Studies

| | Case A: E-commerce checkout A/B test | Case B: Tumour classification |
|---|---|---|
| **Question** | Does a redesigned checkout page improve speed, conversion and revenue? | Can cell-nucleus measurements predict whether a tumour is malignant? |
| **Data** | 2,400 simulated sessions, 6 variables (seed 42) | Breast Cancer Wisconsin (Diagnostic), 569 patients, 30 features |
| **Methods** | Welch t-test, chi-square tests, one-way ANOVA, Tukey HSD, bootstrap | Logistic regression, KNN, decision tree, random forest, cross-validation, grid search |
| **Report** | Week 3 | Week 4 |

The Week 5 capstone report integrates both case studies and adds new analysis: conversion lift by device, a bootstrap of revenue per session, and a cost-based decision threshold chosen from cross-validated predictions (not from the test set).

## Key Results

**Case A (A/B test)**
- Conversion rose from 11.6% to 16.5% with the new design (p < 0.001), and checkout was about 4.1 seconds faster.
- Basket size did not change, so the revenue gain comes from more buyers.
- Revenue per session rose from ₹182 to ₹263 (95% bootstrap interval of the gain: ₹35 to ₹129).
- Conversion depends on device (mobile lowest), and first-time customers spend significantly less per order than returning customers.

**Case B (classifier)**

| Model | Accuracy | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.965 | 0.929 | 0.951 | 0.996 |
| K-Nearest Neighbours | 0.939 | 0.857 | 0.911 | 0.982 |
| Decision Tree | 0.930 | 0.881 | 0.902 | 0.972 |
| Random Forest | 0.965 | 0.905 | 0.950 | 0.994 |

Across 10 random splits, logistic regression's accuracy averaged about 97% (standard deviation about 1.5 points). With a cost-based threshold of 0.18 (missed cancer weighted 5 to 10 times worse than a false alarm), missed cancers on the test set fell from 3 to 1 for 3 additional false alarms.

## Strategic Recommendations (summary)

1. Roll out the new checkout to all traffic and track revenue per session, with a small holdback group to confirm the lift persists.
2. Run a dedicated mobile checkout follow-up test, since mobile has the most traffic and the lowest conversion.
3. Launch a first-to-second purchase programme for new customers.
4. Use the classifier only as clinical decision support, with the threshold set from a cost ratio agreed with clinicians.
5. Validate externally before any real-world use.

The full evidence, impact scenarios and roadmap are in the Week 5 report.

## Repository Structure

```
.
├── analysis.py                                   # Case A: data generation, hypothesis tests, figures
├── ecommerce_ab_test_dataset.csv                 # Case A dataset
├── ml_pipeline.py                                # Case B: preprocessing, models, evaluation, figures
├── breast_cancer_dataset.csv                     # Case B dataset
├── capstone.py                                   # Week 5 extensions: device lift, revenue bootstrap, cost-based threshold
├── Week3_Statistical_Analysis_Hypothesis_Testing.docx
├── Week4_ML_Model_Development_Evaluation.docx
├── Week5_Comprehensive_Project_Report.docx
└── README.md
```

## How to Run

Requires Python 3.9 or later.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install pandas numpy scipy scikit-learn matplotlib

# 3. run the scripts in this order
python analysis.py       # creates the e-commerce dataset, test results and figures
python ml_pipeline.py    # runs the machine learning pipeline
python capstone.py       # needs ecommerce_ab_test_dataset.csv (created by analysis.py)
```

Each script writes its figures (PNG files) and a results JSON file to the working directory. All random seeds are fixed (`42`), so results should be reproducible, although small differences can appear across library versions.

## Methodology Highlights

- Hypotheses and the significance level (α = 0.05) were stated before testing, and assumptions were checked (Shapiro–Wilk, Levene, Q-Q plots) with non-parametric cross-checks.
- Effect sizes and confidence or bootstrap intervals are reported alongside p-values.
- For machine learning, the test set was held out until the final evaluation. Scaling is inside a scikit-learn `Pipeline` to prevent data leakage, and model selection and tuning used stratified 5-fold cross-validation on the training set only.
- Metrics were matched to the cost of errors (recall, F1, ROC AUC), not accuracy alone.

## Limitations and Future Work

- Simulated e-commerce data and a small, single-source medical dataset; no external validation.
- Subgroup (device, segment) samples are small, so those estimates are wide and observational.
- Test set of 114 patients gives about ±1.5 points of uncertainty.
- Future work: real experiment logs, longer tests with a holdback group, nested cross-validation, probability calibration, SHAP explanations, and multi-site validation.

## Author

Lokesh, Yuva Intern programme.
