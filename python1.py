import pandas as pd
import numpy as np
from scipy.stats import chi2, chi2_contingency

age_groups = [
    "0 9", "10 19", "20 29", "30 39",
    "40 49", "50 59", "60+"
]

n_patients = [ 80,  90, 100, 110,  95,  85,  70]   # Σ = 630

deaths = [ 8,  9, 10, 12,  9,  8,  7]            # Σ = 63 (10 % от 630)

survivors = [n - d for n, d in zip(n_patients, deaths)]

df = pd.DataFrame({
    "Всего": n_patients,
    "Умерло": deaths,
    "Выжило": survivors
}, index=age_groups)

print("Исходные данные")
print(df)

expected_death_rate = 0.10
df["Ож. умерло"] = df["Всего"] * expected_death_rate
df["Ож. выжило"] = df["Всего"] - df["Ож. умерло"]

print("\nОжидаемые частоты")
print(df[["Ож. умерло", "Ож. выжило"]])

min_expected = df[["Ож. умерло", "Ож. выжило"]].min().min()

print(f"\n\tМинимальное ожидаемое значение = {min_expected:.2f} (должно быть ≥5)")

def chi2_component(observed, expected):
   """Возвращает (χ² компонент, p значение) для 2×2 таблицы."""
   contingency = np.array([
        [observed["Умерло"], observed["Выжило"]],
        [df["Умерло"].sum() - observed["Умерло"],
         df["Выжило"].sum() - observed["Выжило"]]
    ])
   chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
   return chi2, p, dof, expected


df[["χ²", "p"]] = df.apply(
    lambda row: pd.Series({
        "χ²": chi2_component(row, None)[0],
        "p": chi2_component(row, None)[1]
    }),
    axis=1
)

print("\nχ² компоненты по группам:")
print(df[["χ²", "p"]])

observed_matrix = df[["Умерло", "Выжило"]].values
expected_matrix = df[["Ож. умерло", "Ож. выжило"]].values

chi2_total, p_total, dof_total, _ = chi2_contingency(
    observed_matrix, 
    correction=False, 
    lambda_="log-likelihood"   
)

print("\nСовокупный χ² тест (7 df)")
print(f"χ² = {chi2_total:.4f}, df = {dof_total}, p = {p_total:.4f}")

df_reduced = 6
p_reduced = 1 - chi2.cdf(chi2_total, df_reduced)

print("\nКорректировка df до 6 (общая летальность фиксирована)")
print(f"χ² = {chi2_total:.4f}, df = {df_reduced}, p = {p_reduced:.4f}")