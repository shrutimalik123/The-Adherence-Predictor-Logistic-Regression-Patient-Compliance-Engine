# 💊 The Adherence Predictor: Logistic Regression Patient Compliance Engine

An interactive Supervised Learning simulation designed to teach **Logistic Regression**, **The Sigmoid Activation Function**, and **Binary Classification Thresholding** from scratch. You play as a Lead Specialty Pharmacy Data Analyst constructing an early-warning system that transforms a continuous patient drug compliance rating (Proportion of Days Covered) into a localized probability score to predict and prevent therapy drop-offs.

## 🎓 Learning Objectives

This project focuses on teaching:
* **Logistic Regression Frameworks:** Differentiating how classification models leverage linear baselines to categorize discrete binary target markers ($0$ or $1$).
* **Sigmoid Function Mechanics:** Using mathematical activation equations to map raw, infinite numeric vectors into strict, actionable probability bounds.
* **Log-Odds Transformations:** Mapping the linear transformation stage ($z = mx + b$) before running probability compression curves.
* **Decision Boundary Selection:** Establishing operational threshold filters to convert decimal probability percentages into hard binary clinical directives.

---

## ✨ Features

* **Population Health Management Scenario:** Contextualizes continuous-to-binary statistical classification within a critical care maintenance and clinical outreach workspace.
* **Transparent Activation Math:** Exposes both intermediate log-odds scores and final percentage-scaled Sigmoid outputs side-by-side during runtime.
* **Interactive Parameter Calibration:** Allows custom slope-weight ($m$) and intercept-bias ($b$) parameter tuning to demonstrate how data lines distort probability curves.
* **Zero-Dependency Core:** Developed completely using native Python data arrays and primitive math utilities—no external matrix or modeling frameworks needed.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/adherence-predictor-logistic.git](https://github.com/YOUR_USERNAME/adherence-predictor-logistic.git)
    cd adherence-predictor-logistic
    ```
2.  **Save the Code:** Save the provided script as `adherence_predictor.py`.
3.  **Run the Script:**
    ```bash
    python adherence_predictor.py
    ```

### 3. Gameplay Instructions
1.  **Analyze Patient Compliance Logs:** Evaluate historical data tracking the continuous Proportion of Days Covered (PDC Index %) alongside past patient outcome flags ($0 = \text{Stable}$, $1 = \text{Discontinued}$).
2.  **Calibrate Your Model Weights:** Enter custom values for slope weight ($m$) and bias intercept ($b$) to build your baseline decision line.
3.  **Monitor Inbound Patient Trajectories:** Watch the model capture an active alert for a borderline patient whose metrics drop to a $62.0\%$ compliance scale.
4.  **Audit the Sigmoid Triage Output:** Review the resulting probability score and see if your configured parameters trigger an automated pharmacist phone intervention.

---

## 🧠 Code Structure Highlights

### The Sigmoid Curve Compression
The engine maps raw linear equations into precise probability ranges using the classic exponential squashing function.

```python
# Sigmoid Activation Equation: p = 1 / (1 + e^-z)
try:
    probability = 1.0 / (1.0 + math.exp(-z))
except OverflowError:
    probability = 0.0 if z < 0 else 1.0

