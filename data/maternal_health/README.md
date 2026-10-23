# Maternal Health Risk Data

Source notebook: https://www.kaggle.com/code/yasserhessein/classification-maternal-health-5-algorithms-ml

The CSV is the notebook's Maternal Health Risk Data input. It contains the
features `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`, and
the target `RiskLevel`. The backend trains a Random Forest at import time and
serves predictions through `POST /api/risk/predict`.

The result is a research/demo risk classification, not a diagnosis. A person
with concerning symptoms should contact a qualified healthcare professional or
local emergency services regardless of the model output.