import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Данные
n  = 1000          # гипотетический размер наблюдений
x  = 7             # фактическое число случаев
lam0 = 3           # ожидаемое (порог)
p0 = lam0 / n

# Z статистика (односторонняя)
z = (x - n*p0) / np.sqrt(n*p0*(1-p0))
p_one = 1 - st.norm.cdf(z)          # p value для H1: p > p0
p_two = 2 * (1 - st.norm.cdf(abs(z)))  # двусторонний (для справки)

print(f"Z = {z:.3f}")
print(f"Односторонний p value = {p_one:.4f}")
print(f"Двусторонний p value = {p_two:.4f}")

k = np.arange(0, 15)                     # интересуемся небольшими k
pmf = st.binom.pmf(k, n, p0)

plt.figure(figsize=(6,3))
plt.bar(k, pmf, color='steelblue')
plt.axvline(x, color='red', linestyle='--', label='Наблюдаемые (7)')
plt.title('Биномиальное распределение при p₀ = 3/1000')
plt.xlabel('Число заболевших')
plt.ylabel('Вероятность')
plt.legend()
plt.tight_layout()
plt.show()

alpha = 0.05
ci_low, ci_high = st.binom.interval(1-alpha, n, p0)   # количество случаев
print(f"95 % CI для числа случаев: [{ci_low}, {ci_high}]")

ci_low_rate  = ci_low / n
ci_high_rate = ci_high / n
print(f"95 % CI частоты: [{ci_low_rate:.5f}, {ci_high_rate:.5f}]")