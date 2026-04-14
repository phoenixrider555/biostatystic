import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Данные
lam0 = 250          # ожидаемое
x    = 297          # наблюдаемое

# Z статистика (односторонняя)
z = (x - lam0) / np.sqrt(lam0)
p_one_side = 1 - st.norm.cdf(z)          # p value для H1: λ > 250
p_two_side = 2 * (1 - st.norm.cdf(abs(z)))  # если бы проверяли двусторонне

print(f"Z = {z:.3f}")
print(f"Односторонний p value = {p_one_side:.4f}")
print(f"Двусторонний p value = {p_two_side:.4f}")
x_vals = np.arange(lam0-4*np.sqrt(lam0), lam0+4*np.sqrt(lam0), 0.1)
norm_pdf = st.norm.pdf(x_vals, loc=lam0, scale=np.sqrt(lam0))

plt.figure(figsize=(6,3))
plt.plot(x_vals, norm_pdf, label='N(λ₀, √λ₀)')
plt.axvline(x, color='red', linestyle='--', label='Наблюдаемое (297)')
plt.title('Проверка превышения эпидемического порога')
plt.xlabel('Число заболевших')
plt.ylabel('Плотность')
plt.legend()
plt.tight_layout()
plt.show()

alpha = 0.05
z_crit = st.norm.ppf(1 - alpha/2)   # 1.95996
se = np.sqrt(x)                     # √x
ci_low  = x - z_crit * se
ci_high = x + z_crit * se

print(f"95 % CI: [{ci_low:.1f}, {ci_high:.1f}]")

lam_range = np.arange(0, 500, 0.5)
likelihood = st.poisson.pmf(x, lam_range)

plt.figure(figsize=(6,3))
plt.plot(lam_range, likelihood, label='P(X=297|λ)')
plt.axvline(ci_low, color='green', linestyle='--', label='Нижняя граница')
plt.axvline(ci_high, color='green', linestyle='--', label='Верхняя граница')
plt.title('По точное правдоподобие λ (Пуассон)')
plt.xlabel('λ')
plt.ylabel('Вероятность')
plt.legend()
plt.tight_layout()
plt.show()

x_march = 297
x_april = 313
diff    = x_april - x_march          # 16
se_diff = np.sqrt(x_march + x_april) # sqrt(610)

z_diff = diff / se_diff
p_two_side_diff = 2 * (1 - st.norm.cdf(abs(z_diff)))

print(f"Разность = {diff}")
print(f"Z = {z_diff:.3f}")
print(f"p value (двусторонний) = {p_two_side_diff:.4f}")

d_vals = np.linspace(-4*se_diff, 4*se_diff, 400)
norm_pdf = st.norm.pdf(d_vals, loc=0, scale=se_diff)

plt.figure(figsize=(6,3))
plt.plot(d_vals, norm_pdf, label='N(0, √(λ₁+λ₂))')
plt.axvline(diff, color='red', linestyle='--', label='Наблюдаемая разность (16)')
plt.title('Проверка различия между мартом и апрелем')
plt.xlabel('Разность (апрель – март)')
plt.ylabel('Плотность')
plt.legend()
plt.tight_layout()
plt.show()
