import pandas as pd
import numpy as np
from scipy.stats import binom, norm
from scipy.stats import chi2_contingency
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Примерные данные из текста
data = {
    "Регион": ["Москва", "Московская обл."],
    "I_100k": [0.400, 1.184],          # интенсивность на 100 000
    "Pop": [11_514_300, 7_092_900],   # численность
}
df = pd.DataFrame(data)
df
# Абсолютные случаи A = I * Pop / 100000
df["A"] = df["I_100k"] * df["Pop"] / 100_000
df

alpha = 0.05
z = norm.ppf(1 - alpha/2)

# Доля p = A / Pop
df["p_hat"] = df["A"] / df["Pop"]

# При больших A – нормальное приближение
df["ci_low_norm"] = df["p_hat"] - z * np.sqrt(df["p_hat"] * (1 - df["p_hat"]) / df["Pop"])
df["ci_up_norm"]  = df["p_hat"] + z * np.sqrt(df["p_hat"] * (1 - df["p_hat"]) / df["Pop"])

# При малых A (A < 5) – точный биномиальный интервал
def binom_ci(row):
    a = int(round(row["A"]))
    n = row["Pop"]
    if a < 5:
        low, up = binom.interval(1 - alpha, n, a/n) / n
        return pd.Series([low, up])
    else:
        return pd.Series([np.nan, np.nan])

df[["ci_low_binom", "ci_up_binom"]] = df.apply(binom_ci, axis=1)
df
K = 100_000
df["I_low_norm"] = df["ci_low_norm"] * K
df["I_up_norm"]  = df["ci_up_norm"]  * K
df["I_low_binom"] = df["ci_low_binom"] * K
df["I_up_binom"]  = df["ci_up_binom"]  * K
df[["Регион", "I_100k", "I_low_norm", "I_up_norm", "I_low_binom", "I_up_binom"]]


# Приведём к «гипотетической» численности ~ 1000
hyp_N = 1151   # Москва, из примера
hyp_M = 709    # Московская обл.

# Округлим абсолютные случаи до целых
A_moscow = int(round(df.loc[0, "A"]))   # 46
A_obl   = int(round(df.loc[1, "A"]))   # 84

# Таблица 2×2
contingency = np.array([
    [A_moscow, hyp_N - A_moscow],
    [A_obl,    hyp_M - A_obl]
])
chi2, p, dof, exp = chi2_contingency(contingency, correction=False)
chi2, p


# Дисперсия A для пуассоновского распределения = A
df["Var_A"] = df["A"]

# Дисперсия интенсивного показателя I:
# Var(I) = Var(A) * (K/N)^2
df["Var_I"] = df["Var_A"] * (K / df["Pop"])**2
df["SD_I"]  = np.sqrt(df["Var_I"])

# Разность I и её стандартная ошибка
diff = df.loc[0, "I_100k"] - df.loc[1, "I_100k"]
se_diff = sqrt(df.loc[0, "Var_I"] + df.loc[1, "Var_I"])
t_stat = diff / se_diff
p_two_tail = 2 * (1 - norm.cdf(abs(t_stat)))
t_stat, p_two_tail

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

# Точки интенсивных показателей
sns.pointplot(x="Регион", y="I_100k", data=df, join=False, capsize=.2, ax=ax,
              errwidth=1.5, ci=None, color="steelblue")

# Добавляем доверительные интервалы (нормальные)
for i, row in df.iterrows():
    ax.errorbar(i, row["I_100k"],
                yerr=[[row["I_100k"] - row["I_low_norm"]],
                      [row["I_up_norm"] - row["I_100k"]]],
                fmt='none', ecolor='gray', capsize=5)

ax.set_ylabel("Интенсивность (на 100 000)")
ax.set_title("Сравнение интенсивной заболеваемости")
plt.show()
