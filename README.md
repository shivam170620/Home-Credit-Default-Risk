# Home Credit Default Risk --- EDA, Feature Engineering & Modeling

A beginner-friendly machine learning project that predicts whether a
loan applicant may have payment difficulties. The project walks through
the workflow from understanding the dataset and performing exploratory
data analysis (EDA) to feature engineering, preprocessing, and comparing
Logistic Regression, XGBoost, and LightGBM.

## Kaggle competition and dataset

-   **Original competition:** [Home Credit Default Risk ---
    Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
-   **Competition evaluation details:** [Official evaluation
    page](https://www.kaggle.com/competitions/home-credit-default-risk/overview/evaluation)
-   **Dataset and downloadable files:** [Kaggle Data
    tab](https://www.kaggle.com/competitions/home-credit-default-risk/data)
-   **Related later competition (different dataset/task setup):** [Home
    Credit --- Credit Risk Model Stability
    (2024)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)

> This project is based on the original **Home Credit Default Risk**
> competition. The later 2024 Model Stability competition is related,
> but it is a separate competition; do not treat its data or evaluation
> metric as interchangeable with this project.

## Problem statement

Predict the probability that an applicant experiences payment
difficulties. This is a **binary classification** problem:

-   `TARGET = 0`: no payment difficulties recorded.
-   `TARGET = 1`: payment difficulties recorded.

The original Kaggle competition evaluates submissions using **ROC-AUC**,
which measures how well a model ranks positive cases above negative
cases. Since the target is imbalanced, accuracy alone is not an adequate
model-selection metric.

## Dataset overview

The main training table is `application_train.csv`. The project also
uses historical information from related tables, which are aggregated to
applicant level before being joined to the main application table.

  -------------------------------------------------------------------------------
  Source table                  What it contains          Typical processing
  ----------------------------- ------------------------- -----------------------
  `application_train.csv`       Main loan applications    Main EDA and
                                and `TARGET`              applicant-level
                                                          features

  `bureau.csv`                  Applicants' previous      Aggregate counts and
                                credits at other          credit summaries by
                                financial institutions    `SK_ID_CURR`

  `previous_application.csv`    Applicants' previous Home Counts of applications,
                                Credit applications       approvals/refusals, and
                                                          summary features

  `credit_card_balance.csv`     Monthly credit-card       Utilization and balance
                                balance history           summaries

  `POS_CASH_balance.csv`        Point-of-sale/cash-loan   Delinquency and
                                monthly history           loan-history summaries

  `installments_payments.csv`   Installment payment       Payment delays,
                                history                   lateness, and shortfall
                                                          summaries
  -------------------------------------------------------------------------------

Kaggle data is not included in this repository/document by default.
Download it from the official [Kaggle Data
tab](https://www.kaggle.com/competitions/home-credit-default-risk/data)
and follow Kaggle's data-use terms. File availability and exact columns
should be checked against the competition download.

## Project workflow

1.  **Understand the data**
    -   Inspect shape, column names, data types, unique IDs, duplicate
        rows, and target values.
    -   Separate numerical, categorical, and binary-like columns.
2.  **Exploratory data analysis (EDA)**
    -   Review descriptive statistics and distributions.
    -   Inspect missing values and high-cardinality categorical
        features.
    -   Examine target imbalance and compare patterns across target
        classes.
    -   Calculate numerical correlations and review highly correlated
        feature pairs.
3.  **Aggregate historical tables**
    -   Group history tables by `SK_ID_CURR` and calculate counts,
        averages, maxima, minima, and other summaries.
    -   Merge applicant-level summaries into the main table.
    -   This prevents a one-to-many history join from multiplying
        application rows.
4.  **Feature engineering**
    -   Create historical application counts and approval/refusal
        counts.
    -   Create credit-card utilization and balance summaries.
    -   Derive POS delinquency indicators/rates.
    -   Derive installment payment delays, late-payment indicators, and
        payment shortfalls.
5.  **Preprocess features**
    -   Map selected binary columns to numeric values.
    -   Represent remaining categorical variables with one-hot encoding.
    -   Fit preprocessing steps on training data only, then transform
        validation data.
6.  **Split data correctly**
    -   Use a stratified train/validation split to preserve the target
        class proportion.
    -   Keep the target out of the feature matrix.
7.  **Train and compare models**
    -   Logistic Regression as an interpretable baseline.
    -   XGBoost and LightGBM as gradient-boosted tree models.
    -   Compare ROC-AUC and average precision / PR-AUC; inspect
        precision, recall, F1, and confusion matrices at selected
        thresholds.
8.  **Choose a decision threshold**
    -   The probability threshold can be changed based on the cost of
        missed defaults versus unnecessary flags.
    -   The default threshold of 0.5 is not automatically the best
        threshold for an imbalanced problem.

## Important findings from this project run

The following values describe the notebook run and can vary if the data,
feature set, or split changes.

-   Final engineered training table: **307,511 rows × 230 columns**
    before the final model encoding stage.
-   Applicant ID: `SK_ID_CURR`.
-   Duplicate rows: **0**; duplicate applicant IDs: **0** in the final
    table checked.
-   Target distribution:
    -   `TARGET = 0`: **282,686 (91.93%)**
    -   `TARGET = 1`: **24,825 (8.07%)**
-   The target is strongly imbalanced, so ROC-AUC and precision-recall
    metrics are more informative than accuracy alone.
-   `AMT_INCOME_TOTAL` is highly right-skewed (mean about **168,798**,
    median about **147,150**, with extreme values), so inspect
    distributions and outliers rather than relying on the mean alone.
-   `OCCUPATION_TYPE` has about **96,391 missing values (31.35%)** in
    the main training data.
-   The notebook one-hot encodes remaining categorical variables and
    creates **140 encoded columns** in that preprocessing run.
-   The final encoded train/validation feature matrix has **355
    columns** in that run.
-   Validation split: **80% train / 20% validation**, stratified by
    target, with `random_state=42`.

## Missing values, encoding, and scaling

-   **Numerical columns for Logistic Regression:** impute missing values
    with the training-set median, then apply `StandardScaler`.
-   **Categorical columns:** fill missing categories with the explicit
    label `Missing`, then one-hot encode with `handle_unknown="ignore"`.
-   **Binary-like columns:** map known values consistently, for example
    `FLAG_OWN_CAR` (`N` → 0, `Y` → 1), `FLAG_OWN_REALTY` (`N` → 0, `Y` →
    1), and `NAME_CONTRACT_TYPE` (`Cash loans` → 1, `Revolving loans`
    → 0) as done in the notebook.
-   **Leakage prevention:** fit imputers, scalers, and encoders only on
    training data. Never use `TARGET` as an input feature.
-   **Tree models:** XGBoost and LightGBM can handle missing numerical
    values natively, so the notebook's tree-model path does not require
    the same scaling step used for Logistic Regression. Categorical
    preprocessing is still required for the encoded feature matrix used
    here.

## Model results recorded in the notebook

  -----------------------------------------------------------------------
  Model                      Validation ROC-AUC        Validation average
                                      (approx.)        precision / PR-AUC
                                                                (approx.)
  ------------------- ------------------------- -------------------------
  Logistic Regression                     0.772                     0.255

  XGBoost                                  0.78                      0.28

  LightGBM                                 0.78                      0.28
  -----------------------------------------------------------------------

These are notebook validation results, **not Kaggle leaderboard
scores**. The tree models performed better than the Logistic Regression
baseline in this run, but the displayed values are approximate and
should be reproduced from the notebook before being used as final
reported results.

### Threshold examples from this run

At a Logistic Regression threshold of 0.70, the notebook recorded
approximately:

-   Precision: **0.269**
-   Recall: **0.392**
-   F1-score: **0.319**
-   Confusion matrix: `[[51262, 5276], [3020, 1945]]`

The notebook also explored thresholds around 0.67--0.68 for comparing F1
across models. A threshold selected on the same validation set can be
optimistically biased; use cross-validation or a separate untouched test
set for a more reliable final estimate.

## Tools and libraries

-   Python
-   pandas and NumPy
-   Matplotlib / Seaborn for EDA plots (where used in the notebook)
-   scikit-learn: train/validation split, imputers, scaling, one-hot
    encoding, Logistic Regression, and metrics
-   XGBoost
-   LightGBM
-   Jupyter Notebook

## How to run

1.  Accept the competition rules and download the data from the
    [official Kaggle Data
    tab](https://www.kaggle.com/competitions/home-credit-default-risk/data).

2.  Place the downloaded CSV files in the location expected by the
    notebook, or update the notebook's file paths.

3.  Install the required packages in your Python environment. For
    example:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
    ```

4.  Open the notebook and run the cells in order:

    ``` bash
    jupyter notebook
    ```

5.  Check that the input paths, package versions, and expected column
    names match your downloaded files before running the full pipeline.

## Interview talking points

-   Why the target is imbalanced and why accuracy is misleading.
-   Why one-to-many historical tables must be aggregated before merging.
-   How group-by aggregations create applicant-level features.
-   The difference between numerical, categorical, and binary columns.
-   Why one-hot encoding is used and how `handle_unknown="ignore"` helps
    with unseen categories.
-   Why median imputation and scaling are used for Logistic Regression.
-   Why tree-based models generally do not need feature scaling and can
    handle numerical missing values natively.
-   The difference between ROC-AUC and PR-AUC / average precision.
-   How the decision threshold changes precision and recall.
-   How to avoid leakage by fitting preprocessing only on the training
    split.
-   Why validation metrics are not the same as Kaggle leaderboard
    scores.

## Limitations and next steps

-   The reported metrics are from one validation split; add stratified
    cross-validation for a more robust estimate.
-   If thresholds are selected by F1, select them without using the
    final test set.
-   Check feature importance and use explainability tools where
    appropriate.
-   Audit for data leakage and ensure historical features would have
    been available at prediction time.
-   Compare feature families through controlled experiments instead of
    adding all features without measuring their contribution.
-   This README summarizes the notebook workflow; the notebook remains
    the source of executable code and exact implementation details.

## References

1.  [Home Credit Default Risk --- official Kaggle
    competition](https://www.kaggle.com/competitions/home-credit-default-risk)
2.  [Home Credit Default Risk --- official evaluation
    page](https://www.kaggle.com/competitions/home-credit-default-risk/overview/evaluation)
3.  [Home Credit Default Risk --- official dataset download
    page](https://www.kaggle.com/competitions/home-credit-default-risk/data)
4.  [Home Credit --- Credit Risk Model Stability (separate 2024
    competition)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)

------------------------------------------------------------------------

**Project purpose:** educational portfolio and interview preparation.
Results should be reproduced and validated before being presented as
production-ready performance.
