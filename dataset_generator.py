import pandas as pd
import numpy as np
import random

np.random.seed(50)
random.seed(50)

num_trucks = 24000

truck_ids = [f"T{str(i).zfill(5)}" for i in range(1, num_trucks + 1)]

age = np.random.randint(0, 15, num_trucks)

mileage = age * np.random.randint(8000, 15000, num_trucks) + np.random.randint(0, 10000, num_trucks)

fuel_efficiency = np.round(7 - (age * 0.25) + np.random.uniform(-0.5, 0.5, num_trucks), 2)
fuel_efficiency = np.clip(fuel_efficiency, 2, 7)

battery_health = np.clip(np.round(100 - (age * 3) + np.random.uniform(-5, 5, num_trucks), 1), 20, 100)

service_delay = np.random.randint(0, 60, num_trucks)

engine_temp = np.round(80 + (age * 1.5) + np.random.uniform(-5, 5, num_trucks), 1)

# ── ADD SENSOR NOISE (Option 2) ────────────────────────────────────────────
# Simulates real-world sensor measurement errors
engine_temp = np.round(engine_temp + np.random.normal(0, 2, num_trucks), 1)
fuel_efficiency = np.round(fuel_efficiency + np.random.normal(0, 0.3, num_trucks), 2)
fuel_efficiency = np.clip(fuel_efficiency, 2, 7)

repair_count = np.random.poisson(lam=age * 0.3, size=num_trucks)

status = []

for i in range(num_trucks):
    risk_score = 0

    if age[i] > 8:
        risk_score += 1
    if mileage[i] > 140000:
        risk_score += 1
    if fuel_efficiency[i] < 4:
        risk_score += 1
    if battery_health[i] < 70 or battery_health[i] > 100:
        risk_score += 1
    if service_delay[i] > 30:
        risk_score += 1
    if engine_temp[i] > 100:
        risk_score += 1

    if risk_score >= 2:
        status.append("High Risk")
    elif risk_score == 1:
        status.append("Service Required Soon")
    else:
        status.append("Truck Healthy")

# ── ADD LABEL NOISE (Option 1) ─────────────────────────────────────────────
# Simulates human labeling errors and edge cases in real data
all_statuses = ["High Risk", "Service Required Soon", "Truck Healthy"]
noise_percentage = 0.05  # 5% of records get a wrong label

for i in range(len(status)):
    if random.random() < noise_percentage:
        current = status[i]
        others = [s for s in all_statuses if s != current]
        status[i] = random.choice(others)

df = pd.DataFrame({
    "Truck_ID":        truck_ids,
    "Age":             age,
    "Total_KM_Driven": mileage,
    "Fuel_Efficiency": fuel_efficiency,
    "Battery_Health":  battery_health,
    "Service_Delay":   service_delay,
    "Engine_Temp":     engine_temp,
    "Repair_Count":    repair_count,
    "Status":          status
})

pd.set_option('display.max_columns', None)
print(df.head(10))

print("\nStatus Distribution (count):")
print(df['Status'].value_counts())

print("\nStatus Distribution (percentage):")
print(df['Status'].value_counts(normalize=True).round(2) * 100)

df.to_csv("fleet_health.csv", index=False, mode='w')
print("\nCSV saved successfully!")