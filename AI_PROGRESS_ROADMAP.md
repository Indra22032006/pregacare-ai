# PregaCare AI Progress and Roadmap

## Current status

PregaCare is a synthetic maternal wellness monitoring prototype. It includes
a Random Forest classifier trained from the supplied Maternal Health Risk
dataset, alongside rule-based dashboard warnings. The classifier maps six
measurements to dataset risk labels; it does not forecast clinical outcomes,
diagnose disease, or predict medical complications.

The current application demonstrates the data flow that a future AI system could use:

1. Generate synthetic patient vital readings.
2. Associate readings with individual patients.
3. Display readings on a browser dashboard.
4. Detect simple safety events, such as a possible fall.
5. Highlight readings that cross predefined health thresholds.
6. Classify maternal-health risk from six measurements using a Random Forest
   trained on the supplied dataset.

## What principle is being used now?

### 1. Sensor-data simulation

The simulator produces artificial readings for:

- Heart rate
- SpO2
- Stress percentage
- Temperature
- Systolic and diastolic blood pressure
- Accelerometer movement

The values are bounded so they remain within plausible demonstration ranges. Readings transition gradually rather than changing completely at random. This imitates the type of time-series data that could eventually come from wearable sensors or a Raspberry Pi device.

### 2. Rule-based monitoring

The dashboard compares each reading against predefined thresholds. Depending on the severity, a metric card can be shown as:

- Normal
- Yellow: needs attention
- Orange: concerning
- Red: critical

This is a rules engine, not machine learning. For example, a low SpO2 value can cause an orange or red warning, while a high temperature can cause a critical warning.

### 3. Patient risk profiles

The prototype includes five synthetic patients. Three demonstrate different risk states:

- Aisha Khan: orange risk because of depleted SpO2.
- Nora Johnson: red risk because of a high fever.
- Leila Brown: yellow risk because of elevated stress.

These profiles are deliberately configured for a presentation demo. They are not real patient records.

### 4. Fall detection

The accelerometer values are combined into an acceleration magnitude:

```text
magnitude = sqrt(x^2 + y^2 + z^2)
```

If the magnitude crosses the configured fall threshold, the dashboard displays a fall alert. The Simulate fall button is included to demonstrate this event safely with synthetic data.

## Why this is useful for an AI presentation

The current classifier and rules provide a limited research/demo baseline and
establish parts of the foundation for a future monitoring system:

- A consistent data format
- Patient-specific time-series readings
- A live API
- A dashboard for clinicians or researchers
- Clearly labelled events and outcomes
- Synthetic data for development without exposing real patient information

The rule-based version also provides a baseline. A future AI model should be compared against these simple rules to determine whether it improves early warning accuracy without producing too many false alarms.

## Proposed AI roadmap

### Stage 1: Monitoring prototype - current

- Generate synthetic readings.
- Display five patient dashboards.
- Apply transparent health thresholds.
- Demonstrate fall detection.
- Show color-coded warnings.
- Train a Random Forest from the maternal-health dataset and expose risk-classification endpoints.

### Stage 2: Better synthetic data

- Add realistic patient-specific baselines.
- Generate longer time-series histories.
- Model daily patterns, rest, movement, and measurement noise.
- Add labelled events such as gradual SpO2 decline, fever, elevated heart rate, and falls.
- Store readings in a database instead of generating them only in memory.

### Stage 3: Data validation and preprocessing

- Detect missing readings and sensor failures.
- Remove impossible values and obvious sensor glitches.
- Normalize readings against each patient's personal baseline.
- Calculate trends, rolling averages, rate of change, and duration above or below a threshold.

### Stage 4: Predictive vital-sign models

Train models on historical time-series data to estimate what may happen next. Possible approaches include:

- A baseline statistical model for short-term forecasting.
- Random Forest or Gradient Boosting models using recent vital-sign features.
- Anomaly detection for unusual combinations of readings.
- Time-series neural networks only if the dataset becomes large enough to justify them.

The model could estimate a future risk such as:

```text
risk of SpO2 entering a dangerous range within the next 15 to 30 minutes
```

The system should report both the risk score and the evidence behind it, such as a falling SpO2 trend, increased heart rate, and reduced movement.

### Stage 5: Early-warning system

Instead of waiting until a vital sign becomes critical, the system could warn a doctor when a patient's trend suggests deterioration.

Example workflow:

1. The system receives readings every few minutes.
2. It compares the current values with the patient's baseline.
3. It calculates trends and a short-term predicted risk.
4. If the risk remains high for a defined period, it creates an alert.
5. The alert explains which measurements caused it.
6. A clinician reviews the alert and decides what action is appropriate.

A warning should not automatically diagnose the patient or make a treatment decision.

### Stage 6: Evaluation and safety

- Split data by patient during evaluation to test generalization.
- Measure sensitivity, specificity, precision, recall, and false-alert rate.
- Test early-warning lead time: how far in advance the system detects deterioration.
- Compare the AI model with the current rule-based baseline.
- Log model versions and explanations for each alert.
- Test unusual, missing, delayed, and noisy sensor data.
- Keep a human clinician in the decision loop.

## Example future alert

```text
Patient P001 may enter a low-SpO2 risk range within the next 20 minutes.

Evidence:
- SpO2 decreased from 96% to 92% over 15 minutes.
- Heart rate increased from 82 to 101 bpm.
- The downward SpO2 trend persisted across five readings.

Action: clinician review recommended.
```

This is a prediction and an early-warning signal, not a diagnosis.

## Important limitations

This project is a student prototype using synthetic data. Its readings, risk labels, and alerts must not be used for medical decisions. A real clinical system would require validated medical devices, real clinical datasets, privacy controls, cybersecurity, clinical review, regulatory approval, and extensive testing before use with patients.
