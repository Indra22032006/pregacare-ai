"""
PregaCare AI — Backend Entry Point
Stage 1: Basic FastAPI app with a health-check endpoint and one test endpoint.

Run this file with:
    uvicorn main:app --reload

This is a STUDENT PROTOTYPE. Nothing in this backend diagnoses disease,
predicts clinical outcomes, or replaces a doctor. All scores/labels are
demonstration outputs only.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from datetime import datetime, timezone

if __package__:
    from .simulator.sensor_simulator import generate_reading
    from .ml.maternal_risk_model import model_info, predict_risk
else:
    from simulator.sensor_simulator import generate_reading
    from ml.maternal_risk_model import model_info, predict_risk

# -----------------------------------------------------------------------
# App metadata — this is what shows up on the Swagger docs page (/docs)
# -----------------------------------------------------------------------
app = FastAPI(
    title="PregaCare AI Backend",
    description=(
        "Maternal wellness monitoring & AI assistance platform — "
        "student prototype. Provides synthetic-data-driven wellness "
        "indicators, NOT medical diagnoses."
    ),
    version="0.1.0",
)

# -----------------------------------------------------------------------
# CORS — allows the Flutter frontend (running on a different port/origin)
# to talk to this backend later. Wide open for now during development;
# we will lock this down before anything resembling production.
# -----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # TODO: restrict this later to the Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PATIENTS = [
    {"id": "P001", "name": "Patient A", "week": 18, "risk": "orange", "risk_note": "Low SpO2"},
    {"id": "P002", "name": "Patient B", "week": 24},
    {"id": "P003", "name": "Patient C", "week": 29},
    {"id": "P004", "name": "Patient D", "week": 32, "risk": "red", "risk_note": "High fever"},
    {"id": "P005", "name": "Patient E", "week": 36, "risk": "yellow", "risk_note": "Elevated stress"},
]


@app.get("/")
def root():
    """
    Basic landing route — just confirms the API is reachable at all.
    """
    return {
        "message": "PregaCare AI backend is running.",
        "docs": "/docs",
    }


@app.get("/api/readings")
def sensor_reading(demo_fall: bool = False):
    """Return one synthetic sensor reading for the dashboard."""
    return generate_reading(patient_id="P001", demo_fall=demo_fall)


@app.get("/api/patients")
def patients():
    """Return the five synthetic patients available in the demo."""
    return PATIENTS


@app.get("/api/patients/{patient_id}/readings")
def patient_reading(patient_id: str, demo_fall: bool = False):
    """Return a reading for one selected synthetic patient."""
    if not any(patient["id"] == patient_id for patient in PATIENTS):
        raise HTTPException(status_code=404, detail="Patient not found")
    return generate_reading(patient_id=patient_id, demo_fall=demo_fall)


@app.get("/api/risk/model")
def risk_model_info():
    """Return the trained model's features, classes, and holdout metrics."""
    return model_info()


@app.post("/api/risk/predict")
def risk_prediction(values: dict[str, float]):
    """Classify maternal risk from the six Kaggle dataset measurements."""
    required = {"Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"}
    missing = sorted(required - values.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {', '.join(missing)}")
    try:
        return predict_risk(values)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid measurement: {error}") from error


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the patient selector dashboard."""
    return PATIENT_INDEX_HTML


@app.get("/dashboard/{patient_id}", response_class=HTMLResponse)
def patient_dashboard(patient_id: str):
    """Serve the dashboard for one selected patient."""
    patient = next((item for item in PATIENTS if item["id"] == patient_id), None)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return DASHBOARD_HTML.replace("__PATIENT_ID__", patient["id"]).replace(
        "__PATIENT_NAME__", patient["name"]
    ).replace("__PATIENT_WEEK__", str(patient["week"])).replace(
        "__RISK__", patient.get("risk", "normal")
    ).replace("__RISK_NOTE__", patient.get("risk_note", "No flagged component"))


@app.get("/api/health")
def health_check():
    """
    Health-check endpoint.

    Later, other services (the Flutter app, the simulator, monitoring
    tools) will call this to confirm the backend is alive before trying
    to send data to it.
    """
    return {
        "status": "ok",
        "service": "pregacare-ai-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/test")
def test_endpoint():
    """
    A simple test endpoint with no real logic yet.

    This exists purely so we can confirm, in Stage 1, that:
    1. The server starts correctly.
    2. Routing works.
    3. JSON responses are returned in the shape we expect.

    In later stages this will be replaced by real endpoints like
    POST /api/readings, GET /api/analytics, etc.
    """
    return {
        "message": "Test endpoint working correctly.",
        "stage": 1,
        "next_stage": "Synthetic sensor-data generator (sensor_simulator.py)",
    }


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>__PATIENT_NAME__ - PregaCare Dashboard</title>
    <style>
        :root { color-scheme: light; font-family: system-ui, sans-serif; }
        body { margin: 0; background: #edf2f2; color: #173333; }
        main { max-width: 980px; margin: 0 auto; padding: 32px 20px; }
        header { display: flex; justify-content: space-between; gap: 16px; align-items: end; }
        h1 { margin: 0 0 6px; font-size: 28px; }
        .muted { color: #607575; margin: 0; }
        button { border: 0; border-radius: 6px; padding: 11px 14px; background: #146c69;
            color: white; cursor: pointer; font-weight: 700; }
        button.active { background: #b93838; }
        #alert { margin: 22px 0; padding: 14px 16px; border-radius: 6px; background: #dcebea; }
        #alert.fall { background: #ffd9d6; color: #8a1d1d; font-weight: 700; }
        #prediction { margin: 22px 0; padding: 18px; border: 1px solid #d4dfde; border-radius: 6px; background: white; }
        #prediction.high-risk { background: #ffd9d6; border-color: #c43b35; }
        #prediction.mid-risk { background: #fff8d6; border-color: #e3bd27; }
        #prediction.low-risk { background: #dcebea; border-color: #72aaa7; }
        .inputs { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-top: 14px; }
        label { color: #607575; font-size: 13px; display: grid; gap: 5px; }
        input { width: 110px; border: 1px solid #b7c9c7; border-radius: 4px; padding: 9px; font: inherit; }
        #predictionTitle { margin: 0; font-size: 19px; }
        #predictionText { margin: 7px 0 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
        .card { background: white; border: 1px solid #d4dfde; border-radius: 6px; padding: 18px; }
        .card.status-yellow { background: #fff8d6; border-color: #e3bd27; }
        .card.status-orange { background: #ffead2; border-color: #e48622; }
        .card.status-red { background: #ffd9d6; border-color: #c43b35; }
        .risk-badge { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; vertical-align: middle; text-transform: uppercase; }
        .risk-orange { background: #ffead2; color: #8b4a0b; }
        .risk-red { background: #ffd9d6; color: #8a1d1d; }
        .risk-yellow { background: #fff8d6; color: #725c00; }
        .label { color: #607575; font-size: 13px; }
        .value { display: block; margin-top: 8px; font-size: 25px; font-weight: 750; }
        .wide { grid-column: span 2; }
        @media (max-width: 520px) { header { display: block; } button { margin-top: 16px; } .wide { grid-column: span 1; } }
    </style>
</head>
<body>
    <main>
        <header>
            <div><p class="muted"><a href="/dashboard">All patients</a> | __PATIENT_ID__</p><h1>__PATIENT_NAME__ <span class="risk-badge risk-__RISK__">__RISK__ risk</span></h1><p class="muted">Pregnancy week __PATIENT_WEEK__ · __RISK_NOTE__ · <span id="updated">Waiting for readings...</span></p></div>
            <button id="fallButton" type="button">Simulate fall</button>
        </header>
        <div id="alert">Monitoring synthetic data</div>
        <section id="prediction" aria-live="polite">
            <h2 id="predictionTitle">Medical-attention prediction</h2>
            <p id="predictionText">Enter age and blood sugar to enable the model.</p>
            <div class="inputs">
                <label>Age <input id="ageInput" type="number" min="12" max="65" value="28"></label>
                <label>Blood sugar <input id="bloodSugarInput" type="number" min="0" max="30" step="0.1" value="7"></label>
            </div>
        </section>
        <section class="grid" aria-live="polite">
            <article class="card" id="heartRateCard"><span class="label">Heart rate</span><strong class="value" id="heartRate">--</strong></article>
            <article class="card" id="spo2Card"><span class="label">SpO2</span><strong class="value" id="spo2">--</strong></article>
            <article class="card" id="stressCard"><span class="label">Stress</span><strong class="value" id="stress">--</strong></article>
            <article class="card" id="temperatureCard"><span class="label">Temperature</span><strong class="value" id="temperature">--</strong></article>
            <article class="card" id="bloodPressureCard"><span class="label">Blood pressure</span><strong class="value" id="bloodPressure">--</strong></article>
            <article class="card wide" id="accelerationCard"><span class="label">Accelerometer magnitude</span><strong class="value" id="acceleration">--</strong></article>
        </section>
    </main>
    <script>
        const fallButton = document.getElementById("fallButton");
        const alertBox = document.getElementById("alert");
        const predictionBox = document.getElementById("prediction");
        const predictionTitle = document.getElementById("predictionTitle");
        const predictionText = document.getElementById("predictionText");
        const ageInput = document.getElementById("ageInput");
        const bloodSugarInput = document.getElementById("bloodSugarInput");
        let demoFall = false;

        function setStatus(cardId, status) {
            const card = document.getElementById(cardId);
            card.classList.remove("status-yellow", "status-orange", "status-red");
            if (status !== "normal") card.classList.add(`status-${status}`);
        }

        function classify(value, yellowLow, yellowHigh, orangeLow, orangeHigh, redLow, redHigh) {
            if (value <= redLow || value >= redHigh) return "red";
            if (value <= orangeLow || value >= orangeHigh) return "orange";
            if (value <= yellowLow || value >= yellowHigh) return "yellow";
            return "normal";
        }

        function classifyLow(value, yellow, orange, red) {
            if (value <= red) return "red";
            if (value <= orange) return "orange";
            if (value <= yellow) return "yellow";
            return "normal";
        }

        function classifyHigh(value, yellow, orange, red) {
            if (value >= red) return "red";
            if (value >= orange) return "orange";
            if (value >= yellow) return "yellow";
            return "normal";
        }

        function worseStatus(first, second) {
            const levels = ["normal", "yellow", "orange", "red"];
            return levels[Math.max(levels.indexOf(first), levels.indexOf(second))];
        }

        async function updatePrediction(reading) {
            const values = {
                Age: Number(ageInput.value),
                SystolicBP: reading.blood_pressure.systolic_mmhg,
                DiastolicBP: reading.blood_pressure.diastolic_mmhg,
                BS: Number(bloodSugarInput.value),
                BodyTemp: reading.temperature_c,
                HeartRate: reading.heart_rate_bpm
            };
            const response = await fetch("/api/risk/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(values)
            });
            if (!response.ok) {
                predictionTitle.textContent = "Medical-attention prediction unavailable";
                predictionText.textContent = "Check age and blood sugar values.";
                return;
            }
            const prediction = await response.json();
            const riskClass = prediction.risk_level.replace(" ", "-");
            predictionBox.classList.remove("high-risk", "mid-risk", "low-risk");
            predictionBox.classList.add(riskClass);
            predictionTitle.textContent = `${prediction.risk_level} · ${Math.round(prediction.risk_probability * 100)}% model confidence`;
            predictionText.textContent = prediction.recommendation;
        }

        async function refresh() {
            const response = await fetch(`/api/patients/__PATIENT_ID__/readings?demo_fall=${demoFall}`);
            const reading = await response.json();
            document.getElementById("heartRate").textContent = `${reading.heart_rate_bpm} bpm`;
            document.getElementById("spo2").textContent = `${reading.spo2_percent}%`;
            document.getElementById("stress").textContent = `${reading.stress_percent}%`;
            document.getElementById("temperature").textContent = `${reading.temperature_c} °C`;
            document.getElementById("bloodPressure").textContent = `${reading.blood_pressure.systolic_mmhg}/${reading.blood_pressure.diastolic_mmhg} mmHg`;
            document.getElementById("acceleration").textContent = `${reading.acceleration_magnitude_g} g`;
            setStatus("heartRateCard", classify(reading.heart_rate_bpm, 60, 100, 50, 120, 40, 140));
            setStatus("spo2Card", classifyLow(reading.spo2_percent, 94, 92, 90));
            setStatus("stressCard", classifyHigh(reading.stress_percent, 60, 80, 90));
            setStatus("temperatureCard", classify(reading.temperature_c, 36, 37.8, 35.5, 38.5, 35, 39));
            const systolicStatus = classify(reading.blood_pressure.systolic_mmhg, 100, 140, 90, 160, 80, 180);
            const diastolicStatus = classify(reading.blood_pressure.diastolic_mmhg, 60, 90, 50, 100, 40, 120);
            setStatus("bloodPressureCard", worseStatus(systolicStatus, diastolicStatus));
            setStatus("accelerationCard", reading.fall_detected ? "red" : classifyHigh(reading.acceleration_magnitude_g, 1.8, 2.2, 3));
            document.getElementById("updated").textContent = `Last reading: ${new Date(reading.timestamp).toLocaleTimeString()}`;
            alertBox.textContent = reading.fall_detected ? "Fall detected: accelerometer threshold exceeded" : "Monitoring synthetic data";
            alertBox.classList.toggle("fall", reading.fall_detected);
            await updatePrediction(reading);
        }

        fallButton.addEventListener("click", () => {
            demoFall = !demoFall;
            fallButton.classList.toggle("active", demoFall);
            fallButton.textContent = demoFall ? "Stop fall simulation" : "Simulate fall";
            refresh();
        });
        ageInput.addEventListener("change", refresh);
        bloodSugarInput.addEventListener("change", refresh);
        refresh();
        setInterval(refresh, 3000);
    </script>
</body>
</html>"""


PATIENT_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PregaCare Patients</title>
    <style>
        :root { font-family: system-ui, sans-serif; color: #173333; }
        body { margin: 0; background: #edf2f2; }
        main { max-width: 980px; margin: 0 auto; padding: 32px 20px; }
        h1 { margin-bottom: 6px; }
        .muted { color: #607575; }
        .patients { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 28px; }
        .patient { display: block; padding: 20px; border: 1px solid #d4dfde; border-left: 7px solid #d4dfde; border-radius: 6px; background: white; color: inherit; text-decoration: none; }
        .patient.orange { border-left-color: #e48622; background: #ffead2; }
        .patient.red { border-left-color: #c43b35; background: #ffd9d6; }
        .patient.yellow { border-left-color: #e3bd27; background: #fff8d6; }
        .patient:hover { border-color: #146c69; box-shadow: 0 4px 14px #17333318; }
        .patient strong { display: block; font-size: 20px; margin-bottom: 8px; }
    </style>
</head>
<body>
    <main>
        <h1>PregaCare patient dashboards</h1>
        <p class="muted">Select one of five synthetic patients to view live wellness readings.</p>
        <section class="patients">
            <a class="patient orange" href="/dashboard/P001"><strong>Aisha Khan</strong>Patient P001 · Week 18 · Orange risk · Low SpO2</a>
            <a class="patient" href="/dashboard/P002"><strong>Maya Patel</strong>Patient P002 · Week 24</a>
            <a class="patient" href="/dashboard/P003"><strong>Sofia Williams</strong>Patient P003 · Week 29</a>
            <a class="patient red" href="/dashboard/P004"><strong>Nora Johnson</strong>Patient P004 · Week 32 · Red risk · High fever</a>
            <a class="patient yellow" href="/dashboard/P005"><strong>Leila Brown</strong>Patient P005 · Week 36 · Yellow risk · Elevated stress</a>
        </section>
    </main>
</body>
</html>"""
