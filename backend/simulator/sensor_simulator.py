"""Synthetic maternal wellness readings for the Stage 1 dashboard."""

from datetime import datetime, timezone
from math import sqrt
from random import uniform


FALL_THRESHOLD = 3.0
_PATIENT_STATES = {}


def _step(value: float, change: float, minimum: float, maximum: float) -> float:
    """Move a value slightly while keeping it inside its sensor range."""
    return max(minimum, min(maximum, value + uniform(-change, change)))


def _new_state(patient_id: str) -> dict:
    state = {
        "heart_rate": uniform(72, 88),
        "spo2": uniform(97, 99.5),
        "stress": uniform(25, 45),
        "temperature": uniform(36.4, 37.2),
        "systolic": uniform(112, 128),
        "diastolic": uniform(68, 82),
        "accelerometer": {"x": 0.0, "y": 0.0, "z": 1.0},
    }
    if patient_id == "P001":
        state["spo2"] = uniform(90.8, 92.5)
    elif patient_id == "P004":
        state["temperature"] = uniform(39.2, 39.7)
    elif patient_id == "P005":
        state["stress"] = uniform(61, 68)
    return state


def generate_reading(patient_id: str = "P001", demo_fall: bool = False) -> dict:
    """Return one smoothly changing synthetic reading for a patient."""
    state = _PATIENT_STATES.setdefault(patient_id, _new_state(patient_id))
    state["heart_rate"] = _step(state["heart_rate"], 2.5, 45, 150)
    spo2_min, spo2_max = (90, 93) if patient_id == "P001" else (85, 100)
    stress_min, stress_max = (60, 70) if patient_id == "P005" else (0, 100)
    temperature_min, temperature_max = (39, 40.2) if patient_id == "P004" else (34, 41)
    state["spo2"] = _step(state["spo2"], 0.25, spo2_min, spo2_max)
    state["stress"] = _step(state["stress"], 1.5, stress_min, stress_max)
    state["temperature"] = _step(state["temperature"], 0.05, temperature_min, temperature_max)
    state["systolic"] = _step(state["systolic"], 3, 75, 190)
    state["diastolic"] = _step(state["diastolic"], 2, 40, 125)

    if demo_fall:
        accelerometer = {
            "x": round(uniform(-4.5, 4.5), 2),
            "y": round(uniform(-4.5, 4.5), 2),
            "z": round(uniform(-4.5, 4.5), 2),
        }
        # Ensure the demo event always crosses the detection threshold.
        accelerometer["x"] = 3.5
    else:
        accelerometer = {
            "x": round(_step(state["accelerometer"]["x"], 0.15, -1.2, 1.2), 2),
            "y": round(_step(state["accelerometer"]["y"], 0.15, -1.2, 1.2), 2),
            "z": round(_step(state["accelerometer"]["z"], 0.12, 0.6, 1.4), 2),
        }
        state["accelerometer"] = accelerometer

    acceleration_magnitude = sqrt(
        sum(axis_value**2 for axis_value in accelerometer.values())
    )

    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate_bpm": round(state["heart_rate"], 1),
        "spo2_percent": round(state["spo2"], 1),
        "stress_percent": round(state["stress"], 1),
        "temperature_c": round(state["temperature"], 1),
        "blood_pressure": {
            "systolic_mmhg": round(state["systolic"]),
            "diastolic_mmhg": round(state["diastolic"]),
        },
        "accelerometer": accelerometer,
        "acceleration_magnitude_g": round(acceleration_magnitude, 2),
        "fall_detected": acceleration_magnitude >= FALL_THRESHOLD,
        "demo_fall": demo_fall,
    }
