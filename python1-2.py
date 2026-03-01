import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

data = {
    "ВГА": [323, 412, 245, 199, 231, 94, 175, 199],
    "ВГВ": [143, 251, 125, 91, 99, 52, 108, 74],
    "ВГС": [35, 49, 33, 72, 59, 31, 61, 47],
}
districts = ["ВАО", "ЦАО", "СВАО", "ЮВАО", "ЮЗАО", "ЮАО", "ЗАО", "СЗАО"]

df = pd.DataFrame(data, index=districts)
df["Всего"] = df.sum(axis=1)               # строковые суммы
df.loc["Всего"] = df.sum(axis=0)           # столбцовые суммы, включая "Всего"
df.loc["Всего", "Всего"] = df["Всего"].sum()   # общая сумма = 3208

print("Таблица наблюдаемых частот")
print(df)

total_patients = df.loc["Всего", "Всего"]
prop = df.loc["Всего", ["ВГА", "ВГВ", "ВГС"]] / total_patients
print("\nДоли по видам гепатита (для всей популяции)")
print(prop)

expected = df.loc[districts, "Всего"].values[:, None] * prop.values
expected_df = pd.DataFrame(expected, index=districts, columns=["ВГА", "ВГВ", "ВГС"])
expected_df["Всего"] = expected_df.sum(axis=1)

print("\nОжидаемые частоты")
print(expected_df.round(2))

observed = df.loc[districts, ["ВГА", "ВГВ", "ВГС"]].values
expected_arr = expected_df.loc[districts, ["ВГА", "ВГВ", "ВГС"]].values

chi2, p, dof, _ = chi2_contingency(observed, correction=False)
print("\nРезультат χ² теста")
print(f"χ² = {chi2:.4f}")
print(f"df = {dof}   (должно быть (8 1)*(3 1)=14)")
print(f"p  = {p:.6f}")

min_expected = expected_arr.min()
print(f"\nМинимальное ожидаемое значение = {min_expected:.2f} (должно быть ≥ 5)")


chi2_components = (observed - expected_arr) ** 2 / expected_arr
chi2_components_df = pd.DataFrame(chi2_components,
                                   index=districts,
                                   columns=["ВГА", "ВГВ", "ВГС"])
print("\nКомпоненты χ² по ячейкам")
print(chi2_components_df.round(4))
print("\nСумма всех компонентов = χ² (проверка):",
      chi2_components_df.values.sum())
