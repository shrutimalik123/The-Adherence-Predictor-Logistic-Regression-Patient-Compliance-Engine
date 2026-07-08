import math

def logistic_regression_adherence_game():
    # 1. Scenario: Specialty Pharmacy Population Health Analytics
    print("--- 💊 THE ADHERENCE PREDICTOR: LOGISTIC REGRESSION ENGINE 💊 ---")
    print("Mission: Estimate the probability that a patient will discontinue chronic therapy.")
    print("Goal: Pass a linear equation through a Sigmoid function to map binary risk.")

    # 2. Historical Training Reference Records (Continuous Feature to Binary Target)
    # Feature (X): Proportion of Days Covered (PDC Index % 0-100)
    # Target (y): 1 = Relapsed/Discontinued, 0 = Remained Stable
    history = [
        {"pdc_index": 45.0, "discontinued": 1},
        {"pdc_index": 60.0, "discontinued": 1},
        {"pdc_index": 75.0, "discontinued": 0},
        {"pdc_index": 90.0, "discontinued": 0},
    ]

    print("\n--- 🖥️ HISTORICAL ADHERENCE MONITORING PATIENT LOGS ---")
    for idx, case in enumerate(history):
        status = "Discontinued/Relapsed" if case["discontinued"] == 1 else "Stable/Compliant"
        print(f"Patient {idx+1}: Refill PDC Index = {case['pdc_index']}% -> Outcome: {status} ({case['discontinued']})")

    # 3. Game Inputs: Setting the Weights (m) and Intercept/Bias (b)
    print("\n--- STEP 1: CALIBRATE THE WEIGHT AND BIAS PARAMETERS ---")
    print("We map continuous variables to a log-odds line: z = (m * X) + b")
    try:
        m = float(input("Enter Weight Slope m (Recommended: -0.2): "))
        b = float(input("Enter Bias Intercept b (Recommended: 13.0): "))
    except ValueError:
        m, b = -0.2, 13.0

    # 4. Incoming Critical Patient Profile
    # A patient's tracker flags them dropping down to a borderline PDC metric of 62.0%
    test_patient_x = 62.0
    print(f"\n--- 🚨 DISPATCH ALARM: CLINICAL TRACKER ALERT ---")
    print(f"Incoming Patient Evaluation -> Refill Compliance Index (X): {test_patient_x}%")

    # 5. The Math - Step A: Compute the Linear Log-Odds (z)
    z = (m * test_patient_x) + b
    print(f"\n--- 🔄 COMPUTING LOGISTIC MODEL PROPAGATION ---")
    print(f"Step 1: Computed Intermediate Log-Odds Score (z): {z:.4f}")

    # 6. The Math - Step B: The Sigmoid Activation Function
    # Sigmoid Formula: p = 1 / (1 + e^-z)
    # This compresses any numeric value from negative infinity to positive infinity down into a strict [0, 1] probability range.
    try:
        probability = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        probability = 0.0 if z < 0 else 1.0
        
    print(f"Step 2: Passed through Sigmoid Activation -> Probability of Relapse (p): {probability:.4%}")

    # 7. Decision Boundary Thresholding (Standard Classification Boundary = 0.5)
    decision_threshold = 0.5
    print(f"Configured System Risk Triage Threshold: {decision_threshold * 100}%")

    if probability >= decision_threshold:
        prediction = 1
        verdict = "⚠️ HIGH RISK FLAG: SYSTEM AUTOMATICALLY SCHEDULES CLINICAL PHARMACIST OUTREACH"
    else:
        prediction = 0
        verdict = "✅ LOW RISK VERDICT: PATIENT REMAINS ON STABLE MAINTENANCE AUTOMATION"

    print(f"\nAutomated Triage Action: {verdict}")

    # 8. Ground Truth Evaluation
    actual_truth = 1 # In real life, a 62% compliance marker frequently triggers therapy failure.
    if prediction == actual_truth:
        print("\n🏆 SUCCESS: Your Logistic Regression model successfully predicted the adherence drop-off!")
        print("Care coordinators can now call the patient to resolve cost or side-effect barriers.")
    else:
        print("\n💥 ANALYTICS INCIDENT: Misclassified risk bounds! A high-risk patient silently slipped through without an alert.")

if __name__ == "__main__":
    logistic_regression_adherence_game()
