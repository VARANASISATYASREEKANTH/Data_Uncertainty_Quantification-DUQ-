**Data Uncertainty Quantification (DUQ)** is the process of identifying,
measuring, and managing the uncertainty that comes from data itself, and
understanding how that uncertainty impacts predictions, models, and
decision-making.

In simple terms: whenever we build a model or analyze data, the data is
never perfect---there may be noise, missing values, sampling bias, or
limited coverage. Data UQ helps quantify *how much trust we can place in
the data-driven results*.

------------------------------------------------------------------------

## 🔑 Key Aspects of Data Uncertainty Quantification

### 1. Sources of Uncertainty in Data

-   **Aleatoric uncertainty**: Natural randomness in the data (e.g.,
    sensor noise, measurement variability).\
-   **Epistemic uncertainty**: Due to lack of knowledge, incomplete
    data, or limited coverage of scenarios.\
-   **Label uncertainty**: Human annotation errors or ambiguity in
    ground-truth labels.\
-   **Distributional uncertainty**: When training and test data come
    from different distributions (domain shift).

### 2. Methods to Quantify Data Uncertainty

-   **Statistical modeling** (confidence intervals, Bayesian inference,
    variance estimation).\
-   **Probabilistic ML models** (Bayesian neural networks, Gaussian
    processes).\
-   **Ensemble methods** (train multiple models and measure variance
    across predictions).\
-   **Monte Carlo dropout** or sampling-based approaches to approximate
    uncertainty.\
-   **Data quality metrics** (noise level estimation, missing data
    imputation uncertainty).

### 3. Applications

-   **Predictive modeling**: Understanding reliability of model
    outputs.\
-   **Risk management**: Making safer decisions in high-stakes domains
    (healthcare, autonomous driving).\
-   **Active learning**: Identifying uncertain samples to prioritize
    labeling.\
-   **Sensor fusion**: Combining multiple uncertain measurements for
    robust estimates.

------------------------------------------------------------------------

✅ **In short:** Data Uncertainty Quantification is about measuring the
"confidence" in data and model predictions by modeling randomness,
noise, and incomplete knowledge---so that decisions made from data are
trustworthy.
