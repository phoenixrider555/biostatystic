import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

n_on = 93         
men_on = 31        
women_on = n_on - men_on

n_off = 52
men_off = 36
women_off = n_off - men_off

obs = pd.DataFrame({
    "Мужчины": [men_on, men_off],
    "Женщины": [women_on, women_off]
}, index=["Очный", "Заочный"])

print("Наблюдаемая таблица:")
print(obs)


def chi2_test(row):
   
    contingency = np.array([
        [row["Мужчины"], row["Женщины"]],
        [obs["Мужчины"].sum() - row["Мужчины"],
         obs["Женщины"].sum() - row["Женщины"]]
    ])
    chi2, p, dof, _ = chi2_contingency(contingency, correction=False)
    return chi2, p

chi2_on, p_on = chi2_test(obs.loc["Очный"])
chi2_off, p_off = chi2_test(obs.loc["Заочный"])

print("\nОтдельные потоки:")
print(f"Очный: χ² = {chi2_on:.4f}, p = {p_on:.4f}")

print(f"Заочный: χ² = {chi2_off:.4f}, p = {p_off:.4f}")

chi2_total, p_total, dof_total, _ = chi2_contingency(obs.values, correction=False)

print("\nСовокупный тест:")
print(f"χ² = {chi2_total:.4f}, df = {dof_total}, p = {p_total:.4f}")

expected = chi2_contingency(obs.values, correction=False)[3] 
min_expected = expected.min()
print("\nУсловия применимости χ² теста:")
print(f"Минимальное ожидаемое значение = {min_expected:.2f} (должно быть ≥ 5)")
print(f"Общее число наблюдений = {obs.values.sum()} (должно быть ≥ 20)")