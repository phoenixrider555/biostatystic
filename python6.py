import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 1. Расчет коэффициента корреляции
print("=" * 80)
print("1. РАСЧЕТ КОЭФФИЦИЕНТА КОРРЕЛЯЦИИ")
print("=" * 80)

# Создадим пример данных для демонстрации
np.random.seed(42)
n = 200

# Имитация данных пациентов
data = {
    'Умер': np.random.binomial(1, 0.3, n),  # 0=выжил, 1=умер
    'Возраст': np.random.normal(65, 15, n),
    'Пол': np.random.binomial(1, 0.5, n),  # 0=женщины, 1=мужчины
    'Сознание': np.random.choice([1, 2], n, p=[0.2, 0.8]),  # 1=спутанное, 2=ясное
    'ЧСС': np.random.normal(85, 20, n),
    'Дыхание': np.random.normal(22, 8, n),
    'Давление': np.random.normal(130, 25, n),
    'Температура': np.random.normal(37.5, 1, n)
}

df = pd.DataFrame(data)
df['ЧСС'] = np.clip(df['ЧСС'], 40, 180)
df['Дыхание'] = np.clip(df['Дыхание'], 8, 40)
df['Давление'] = np.clip(df['Давление'], 80, 200)
df['Возраст'] = np.clip(df['Возраст'], 18, 100)

# Добавим корреляции для реалистичности
df.loc[df['Умер'] == 1, 'ЧСС'] += np.random.normal(15, 5, sum(df['Умер'] == 1))
df.loc[df['Умер'] == 1, 'Дыхание'] += np.random.normal(5, 2, sum(df['Умер'] == 1))
df.loc[df['Умер'] == 1, 'Давление'] -= np.random.normal(10, 5, sum(df['Умер'] == 1))
df.loc[df['Сознание'] == 1, 'Умер'] = np.random.binomial(1, 0.6, sum(df['Сознание'] == 1))

# Вычисляем матрицу корреляций
correlation_matrix = df.corr()

print("Матрица корреляций:")
print(correlation_matrix.round(3))

# Выводим p-значения
print("\nПроверка значимости корреляций с переменной 'Умер':")
for col in df.columns:
    if col != 'Умер':
        corr, p_value = stats.pearsonr(df['Умер'], df[col])
        significance = "**" if p_value < 0.001 else "*" if p_value < 0.05 else ""
        print(f"{col:20s}: r = {corr:6.3f}, p = {p_value:6.4f} {significance}")
# 2. Построение рисунка совместного распределения
print("\n" + "=" * 80)
print("2. ПОСТРОЕНИЕ ГРАФИКОВ СОВМЕСТНОГО РАСПРЕДЕЛЕНИЯ")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# График 1: ЧСС vs Дыхание
axes[0].scatter(df['ЧСС'], df['Дыхание'], alpha=0.6, edgecolors='w', linewidth=0.5)
axes[0].set_xlabel('ЧСС (уд/мин)')
axes[0].set_ylabel('Частота дыхания (дых/мин)')
axes[0].set_title('ЧСС vs Частота дыхания')
axes[0].grid(True, alpha=0.3)

# График 2: ЧСС vs Дыхание с маркерами по исходу
colors = ['blue' if outcome == 0 else 'red' for outcome in df['Умер']]
axes[1].scatter(df['ЧСС'], df['Дыхание'], c=colors, alpha=0.6, 
                edgecolors='w', linewidth=0.5)
axes[1].set_xlabel('ЧСС (уд/мин)')
axes[1].set_ylabel('Частота дыхания (дых/мин)')
axes[1].set_title('ЧСС vs Частота дыхания (синий=выжил, красный=умер)')
axes[1].grid(True, alpha=0.3)

# График 3: Возраст vs Давление с маркерами по исходу
scatter = axes[2].scatter(df['Возраст'], df['Давление'], c=df['Умер'], 
                          cmap='coolwarm', alpha=0.6, edgecolors='w', linewidth=0.5)
axes[2].set_xlabel('Возраст (лет)')
axes[2].set_ylabel('Систолическое давление (мм рт.ст.)')
axes[2].set_title('Возраст vs Давление')
axes[2].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[2], label='Исход (0=выжил, 1=умер)')

plt.tight_layout()
plt.show()

# 3. Определение достоверности отличия коэффициента корреляции от ожидаемого
print("\n" + "=" * 80)
print("3. ПРОВЕРКА ОТЛИЧИЯ КОЭФФИЦИЕНТА КОРРЕЛЯЦИИ ОТ ОЖИДАЕМОГО")
print("=" * 80)

def test_correlation_difference(r_observed, r_expected, n):
    """Проверка отличия наблюдаемого коэффициента корреляции от ожидаемого"""
    # Преобразование Фишера
    z_observed = np.arctanh(r_observed)
    z_expected = np.arctanh(r_expected)
    
    # Разность
    diff = z_observed - z_expected
    
    # Стандартная ошибка
    se = 1 / np.sqrt(n - 3)
    
    # z-статистика
    z_stat = diff / se
    
    # p-значение (двусторонний тест)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, z_observed, z_expected

# Пример из текста
r_observed = 0.612
r_expected = 0.5
n = 49

z_stat, p_value, z_obs, z_exp = test_correlation_difference(r_observed, r_expected, n)

print(f"Наблюдаемый r: {r_observed}")
print(f"Ожидаемый r: {r_expected}")
print(f"N: {n}")
print(f"z-наблюдаемое: {z_obs:.6f}")
print(f"z-ожидаемое: {z_exp:.6f}")
print(f"Разность z: {z_obs - z_exp:.6f}")
print(f"Стандартная ошибка: {1/np.sqrt(n-3):.6f}")
print(f"z-статистика: {z_stat:.6f}")
print(f"p-значение: {p_value:.6f}")

if p_value < 0.05:
    print("Вывод: Наблюдаемый коэффициент достоверно отличается от ожидаемого")
else:
    print("Вывод: Наблюдаемый коэффициент не отличается от ожидаемого")

# 4. Расчет доверительных интервалов для коэффициента корреляции
print("\n" + "=" * 80)
print("4. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ КОЭФФИЦИЕНТА КОРРЕЛЯЦИИ")
print("=" * 80)

def correlation_ci(r, n, alpha=0.05):
    """Расчет доверительного интервала для коэффициента корреляции"""
    # Преобразование Фишера
    z = np.arctanh(r)
    
    # Стандартная ошибка
    se = 1 / np.sqrt(n - 3)
    
    # z-критическое значение
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Доверительный интервал для z
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # Обратное преобразование
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    
    return r_lower, r_upper, z, se

# Примеры из текста
cases = [
    {"название": "Легкая", "r": 0.612, "n": 49},
    {"название": "Средняя", "r": 0.711, "n": 73},
    {"название": "Тяжелая", "r": 0.881, "n": 94}
]

results = []
for case in cases:
    r_lower, r_upper, z, se = correlation_ci(case["r"], case["n"])
    error_minus = case["r"] - r_lower
    error_plus = r_upper - case["r"]
    
    results.append({
        "Группа": case["название"],
        "r": case["r"],
        "N": case["n"],
        "ДИ нижнее": r_lower,
        "ДИ верхнее": r_upper,
        "Погрешность -": error_minus,
        "Погрешность +": error_plus,
        "z": z,
        "SE": se
    })

results_df = pd.DataFrame(results)
print("Доверительные интервалы для коэффициентов корреляции:")
print(results_df.round(4))

# Визуализация с доверительными интервалами
fig, ax = plt.subplots(figsize=(10, 6))

groups = results_df['Группа'].values
r_values = results_df['r'].values
errors_minus = results_df['Погрешность -'].values
errors_plus = results_df['Погрешность +'].values

ax.errorbar(range(len(groups)), r_values, 
            yerr=[errors_minus, errors_plus],
            fmt='o', capsize=10, capthick=2, markersize=10)

ax.set_xlabel('Тяжесть заболевания')
ax.set_ylabel('Коэффициент корреляции')
ax.set_title('Коэффициенты корреляции с доверительными интервалами (95% ДИ)')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Добавим значения на график
for i, (r, lower, upper) in enumerate(zip(r_values, 
                                          results_df['ДИ нижнее'], 
                                          results_df['ДИ верхнее'])):
    ax.text(i, r + 0.03, f'r = {r:.3f}', ha='center', fontsize=10)
    ax.text(i, r - 0.05, f'({lower:.3f}, {upper:.3f})', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# 5. Определение достоверности различий двух коэффициентов корреляции
print("\n" + "=" * 80)
print("5. СРАВНЕНИЕ ДВУХ КОЭФФИЦИЕНТОВ КОРРЕЛЯЦИИ")
print("=" * 80)

def compare_two_correlations(r1, n1, r2, n2):
    """Сравнение двух независимых коэффициентов корреляции"""
    # Преобразование Фишера
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    
    # Дисперсии
    var1 = 1 / (n1 - 3)
    var2 = 1 / (n2 - 3)
    
    # Разность
    diff = z1 - z2
    
    # Дисперсия разности
    var_diff = var1 + var2
    
    # z-статистика
    z_stat = diff / np.sqrt(var_diff)
    
    # p-значение (двусторонний тест)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, diff, var_diff

# Пример из текста
r_light = 0.612
n_light = 49
r_severe = 0.881
n_severe = 94

z_stat_diff, p_value_diff, diff, var_diff = compare_two_correlations(
    r_light, n_light, r_severe, n_severe
)

print(f"Коэффициент корреляции (легкая): {r_light}, N = {n_light}")
print(f"Коэффициент корреляции (тяжелая): {r_severe}, N = {n_severe}")
print(f"z1 = {np.arctanh(r_light):.6f}, z2 = {np.arctanh(r_severe):.6f}")
print(f"Разность z: {diff:.6f}")
print(f"Дисперсия z1: {1/(n_light-3):.6f}")
print(f"Дисперсия z2: {1/(n_severe-3):.6f}")
print(f"Дисперсия разности: {var_diff:.6f}")
print(f"z-статистика: {z_stat_diff:.6f}")
print(f"p-значение: {p_value_diff:.6f}")

if p_value_diff < 0.05:
    print("Вывод: Коэффициенты корреляции достоверно различаются")
else:
    print("Вывод: Коэффициенты корреляции не различаются")

# 6. Частные коэффициенты корреляции
print("\n" + "=" * 80)
print("6. ЧАСТНЫЕ КОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ")
print("=" * 80)

# Для демонстрации создадим данные с контролируемой переменной
np.random.seed(42)
n_samples = 300

# Создаем данные с зависимостями
age = np.random.normal(60, 15, n_samples)
age = np.clip(age, 18, 90)

# Сознание зависит от возраста
consciousness_prob = 1 / (1 + np.exp(-(age - 70) / 10))
consciousness = np.random.binomial(1, consciousness_prob)

# Дыхание зависит от сознания и случайной составляющей
breathing = 20 + 10 * consciousness + np.random.normal(0, 5, n_samples)

# Исход зависит от дыхания и сознания
outcome_prob = 1 / (1 + np.exp(-(0.1*breathing + 2*consciousness - 5)))
outcome = np.random.binomial(1, outcome_prob)

# Создаем DataFrame для анализа
df_partial = pd.DataFrame({
    'Возраст': age,
    'Сознание': consciousness,
    'Дыхание': breathing,
    'Исход': outcome
})

# Расчет обычной корреляции
corr_simple, p_simple = stats.pearsonr(df_partial['Дыхание'], df_partial['Исход'])

print(f"Обычная корреляция Дыхание-Исход: r = {corr_simple:.4f}, p = {p_simple:.4f}")

# Ручной расчет частной корреляции (контроль за сознанием)
def partial_correlation(x, y, control):
    """Расчет частного коэффициента корреляции"""
    # Регрессия x на control
    slope_x, intercept_x, r_x, p_x, std_err_x = stats.linregress(control, x)
    residuals_x = x - (intercept_x + slope_x * control)
    
    # Регрессия y на control
    slope_y, intercept_y, r_y, p_y, std_err_y = stats.linregress(control, y)
    residuals_y = y - (intercept_y + slope_y * control)
    
    # Корреляция остатков
    corr_partial, p_partial = stats.pearsonr(residuals_x, residuals_y)
    
    return corr_partial, p_partial

# Расчет частной корреляции
corr_partial, p_partial = partial_correlation(
    df_partial['Дыхание'], 
    df_partial['Исход'], 
    df_partial['Сознание']
)

print(f"Частная корреляция Дыхание-Исход (контроль за сознанием): r = {corr_partial:.4f}, p = {p_partial:.4f}")

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# График 1: Дыхание vs Исход
axes[0].scatter(df_partial['Дыхание'], df_partial['Исход'], alpha=0.6)
axes[0].set_xlabel('Частота дыхания')
axes[0].set_ylabel('Исход (0=выжил, 1=умер)')
axes[0].set_title(f'Обычная корреляция\nr = {corr_simple:.3f}')
axes[0].grid(True, alpha=0.3)

# График 2: Дыхание vs Сознание
axes[1].scatter(df_partial['Дыхание'], df_partial['Сознание'], alpha=0.6)
axes[1].set_xlabel('Частота дыхания')
axes[1].set_ylabel('Сознание (0=ясное, 1=спутанное)')
axes[1].set_title('Дыхание vs Сознание')
axes[1].grid(True, alpha=0.3)

# График 3: Остатки регрессий
slope_x, intercept_x, _, _, _ = stats.linregress(df_partial['Сознание'], df_partial['Дыхание'])
residuals_x = df_partial['Дыхание'] - (intercept_x + slope_x * df_partial['Сознание'])

slope_y, intercept_y, _, _, _ = stats.linregress(df_partial['Сознание'], df_partial['Исход'])
residuals_y = df_partial['Исход'] - (intercept_y + slope_y * df_partial['Сознание'])

axes[2].scatter(residuals_x, residuals_y, alpha=0.6)
axes[2].set_xlabel('Остатки Дыхание|Сознание')
axes[2].set_ylabel('Остатки Исход|Сознание')
axes[2].set_title(f'Частная корреляция\nr = {corr_partial:.3f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Сводная таблица сравнения
print("\n" + "=" * 80)
print("СВОДНЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА КОРРЕЛЯЦИЙ")
print("=" * 80)

summary_data = {
    "Анализ": [
        "Корреляция Дыхание-Исход",
        "Частная корреляция (контроль за сознанием)",
        "Сравнение групп (легкая vs тяжелая)",
        "Отличие от ожидаемого (0.612 vs 0.5)"
    ],
    "Коэффициент": [
        f"{corr_simple:.3f}",
        f"{corr_partial:.3f}",
        f"{r_light:.3f} vs {r_severe:.3f}",
        f"{r_observed:.3f} vs {r_expected:.3f}"
    ],
    "p-значение": [
        f"{p_simple:.4f}",
        f"{p_partial:.4f}",
        f"{p_value_diff:.4f}",
        f"{p_value:.4f}"
    ],
    "Вывод": [
        "Значимая" if p_simple < 0.05 else "Незначимая",
        "Значимая" if p_partial < 0.05 else "Незначимая",
        "Различаются" if p_value_diff < 0.05 else "Не различаются",
        "Отличается" if p_value < 0.05 else "Не отличается"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))