import numpy as np
import pandas as pd
import os

# ── Reproducible randomness ───────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Dataset schema (mirrors the Kaggle dataset exactly) ──────────────────────
COLUMNS = [
    "Person ID", "Gender", "Age", "Occupation",
    "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "BMI Category", "Blood Pressure",
    "Heart Rate", "Daily Steps", "Sleep Disorder"
]

OCCUPATIONS = [
    "Nurse", "Doctor", "Engineer", "Lawyer", "Teacher",
    "Accountant", "Salesperson", "Software Engineer", "Scientist", "Manager"
]

BMI_CATEGORIES = ["Normal", "Normal Weight", "Overweight", "Obese"]

SLEEP_DISORDERS = ["None", "Sleep Apnea", "Insomnia"]


def _blood_pressure(bmi_cat: str, age: int, stress: int) -> str:
    """Generate a realistic systolic/diastolic BP string based on risk factors."""
    base_systolic = 110
    base_diastolic = 70

    if bmi_cat in ("Overweight", "Obese"):
        base_systolic += np.random.randint(10, 25)
        base_diastolic += np.random.randint(5, 15)
    if age > 45:
        base_systolic += np.random.randint(5, 20)
    if stress >= 7:
        base_systolic += np.random.randint(5, 15)

    systolic  = int(np.clip(base_systolic  + np.random.normal(0, 6), 90, 180))
    diastolic = int(np.clip(base_diastolic + np.random.normal(0, 4), 60, 120))
    return f"{systolic}/{diastolic}"


def _assign_disorder(bmi_cat: str, stress: int, sleep_dur: float,
                     age: int, gender: str) -> str:
    """
    Assign a sleep disorder using realistic probabilistic rules:
      - Sleep Apnea is more likely for obese/overweight males with short sleep
      - Insomnia is more likely under high stress regardless of gender
    """
    p_apnea   = 0.05
    p_insomnia = 0.05

    if bmi_cat in ("Overweight", "Obese"):
        p_apnea += 0.20
    if gender == "Male":
        p_apnea += 0.08
    if sleep_dur < 6.5:
        p_apnea += 0.10
    if age > 40:
        p_apnea += 0.07

    if stress >= 7:
        p_insomnia += 0.25
    if sleep_dur < 6.0:
        p_insomnia += 0.15

    r = np.random.random()
    if r < p_apnea:
        return "Sleep Apnea"
    elif r < p_apnea + p_insomnia:
        return "Insomnia"
    return "None"


def generate_dataset(n: int = 400) -> pd.DataFrame:
    """
    Synthesise n records that closely match the Kaggle dataset's
    statistical profile and cross-variable correlations.
    """
    np.random.seed(RANDOM_SEED)
    records = []

    for pid in range(1, n + 1):
        gender     = np.random.choice(["Male", "Female"], p=[0.52, 0.48])
        age        = int(np.random.normal(43, 11))
        age        = int(np.clip(age, 27, 59))
        occupation = np.random.choice(OCCUPATIONS)

        # Sleep duration – nurses/doctors skew shorter
        base_dur = 7.0 if occupation not in ("Nurse", "Doctor") else 6.5
        sleep_dur = round(np.clip(np.random.normal(base_dur, 0.7), 5.5, 9.0), 1)

        stress     = int(np.clip(np.random.normal(5.5, 2), 3, 9))
        phys_act   = int(np.clip(np.random.normal(60, 20), 30, 90))
        daily_steps = int(np.clip(np.random.normal(7000, 2500), 3000, 15000))
        heart_rate  = int(np.clip(np.random.normal(70, 8), 55, 95))

        # BMI correlates loosely with physical activity
        if phys_act > 70:
            bmi_cat = np.random.choice(BMI_CATEGORIES, p=[0.45, 0.30, 0.20, 0.05])
        else:
            bmi_cat = np.random.choice(BMI_CATEGORIES, p=[0.20, 0.20, 0.40, 0.20])

        # Sleep quality anti-correlates with stress and short sleep
        base_quality = 8.5 - stress * 0.4 + (sleep_dur - 6) * 0.5
        quality = int(np.clip(round(base_quality + np.random.normal(0, 0.5)), 4, 9))

        bp       = _blood_pressure(bmi_cat, age, stress)
        disorder = _assign_disorder(bmi_cat, stress, sleep_dur, age, gender)

        records.append([
            pid, gender, age, occupation,
            sleep_dur, quality, phys_act,
            stress, bmi_cat, bp,
            heart_rate, daily_steps, disorder
        ])

    return pd.DataFrame(records, columns=COLUMNS)


def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """
    Public API used by other modules.
    - If csv_path points to a real file, load that CSV.
    - Otherwise, generate and cache a synthetic dataset in /tmp.
    Returns a cleaned DataFrame ready for preprocessing.
    """
    cache_path = "/tmp/sleep_health_dataset.csv"

    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # Kaggle CSV uses NaN for people with no disorder; normalise to "None"
        if "Sleep Disorder" in df.columns:
            df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
        print(f"[data_loader] Loaded real dataset from {csv_path} ({len(df)} rows)")
    elif os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        print(f"[data_loader] Loaded cached synthetic dataset ({len(df)} rows)")
    else:
        df = generate_dataset(400)
        df.to_csv(cache_path, index=False)
        print(f"[data_loader] Generated & cached synthetic dataset ({len(df)} rows)")

    return df


# ── Quick sanity check when run directly ─────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print("\nDisorder distribution:\n", df["Sleep Disorder"].value_counts())
    print("\nBMI distribution:\n", df["BMI Category"].value_counts())