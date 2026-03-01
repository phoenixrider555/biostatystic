import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2


raw = {
    "study":  [1, 2, 3, 4, 5, 6, 7, 8],
    "year":   [1997, 2003, 2006, 2006, 2008, 2012, 2017, 2018],
    "n":      [73, 92, 20, 71, 99, 35, 77, 31],
    "success":[61, 77, 14, 55, 88, 29, 66, 24],
}
df = pd.DataFrame(raw)
df["fail"] = df["n"] - df["success"]
df["p"] = df["success"] / df["n"]                     # частота «да»

total_n       = df["n"].sum()
total_success = df["success"].sum()
p_overall     = total_success / total_n               # 0.8313…

df["exp_success"] = df["n"] * p_overall
df["exp_fail"]    = df["n"] * (1 - p_overall)

df["chi_sq_success"] = (df["success"] - df["exp_success"])**2 / df["exp_success"]
df["chi_sq_fail"]    = (df["fail"]    - df["exp_fail"]   )**2 / df["exp_fail"]

chi2_val = df["chi_sq_success"].sum() + df["chi_sq_fail"].sum()
df_chi = df.copy()   # сохраняем для возможного вывода
df_chi["df"] = len(df) - 1
p_val = 1 - chi2.cdf(chi2_val, df_chi["df"].iloc[0])


k = len(df)                     # число групп
N_minus1 = df["n"] - 1

df["var_i"] = df["p"] * (1 - df["p"])

s2_w = (N_minus1 * df["var_i"]).sum() / N_minus1.sum()

p_bar = (df["n"] * df["p"]).sum() / total_n

s2_b = (df["n"] * (df["p"] - p_bar)**2).sum() / (k - 1)

ratio = s2_b / s2_w


z = 1.96
df["ci_low"]  = df["p"] - z * np.sqrt(df["p"] * (1 - df["p"]) / df["n"])
df["ci_up"]   = df["p"] + z * np.sqrt(df["p"] * (1 - df["p"]) / df["n"])


print("\n=== Хи квадрат тест гомогенности ===")
print(f"χ² = {chi2_val:.3f}")
print(f"df = {k-1}")
print(f"p value = {p_val:.3f}")

print("\n=== Дисперсии ===")
print(f"s²_w (внутри групп) = {s2_w:.5f}")
print(f"s²_b (между группами) = {s2_b:.5f}")
print(f"Отношение s²_b / s²_w = {ratio:.3f}")


plt.figure(figsize=(8, 4.5))
plt.errorbar(df["year"], df["p"], 
             xerr=0, yerr=[df["p"]-df["ci_low"], df["ci_up"]-df["p"]],
             fmt='o', capsize=4, ecolor='gray', color='steelblue')
plt.title("Частота успешного лечения по годам")
plt.xlabel("Год публикации")
plt.ylabel("Доля успеха")
plt.ylim(0, 1)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()


df_sorted = df.sort_values("year")
df_sorted["cum_success"] = df_sorted["success"].cumsum()
df_sorted["cum_n"]       = df_sorted["n"].cumsum()
df_sorted["cum_rate"]    = df_sorted["cum_success"] / df_sorted["cum_n"]

plt.figure(figsize=(8, 4.5))
plt.step(df_sorted["year"], df_sorted["cum_rate"], where='post',
         label='Накопительная частота', color='darkorange')
plt.scatter(df_sorted["year"], df_sorted["cum_rate"], color='darkorange')
plt.title("Накопительная частота успеха (елочка)")
plt.xlabel("Год")
plt.ylabel("Кумулятивная доля успеха")
plt.ylim(0, 1)
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()