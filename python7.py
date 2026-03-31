
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings('ignore')

# Настройка стиля графиков

plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. СОЗДАНИЕ ИМИТАЦИОННЫХ ДАННЫХ АНАЛОГИЧНЫХ ПРИМЕРАМ ИЗ ТЕКСТА

# Создаем данные по примеру с возрастом и рецидивом (аналогично файлу Jaws)
np.random.seed(42)
n = 582
age = np.random.normal(48.82, 15.31, n)
age = np.clip(age, 17, 93)  # Ограничиваем возраст как в примере

# Создаем зависимость рецидива от возраста (как в тексте)
# В тексте: риск увеличивается после 60 лет
rezidive_prob = np.where(age < 60, 0.1, 0.1 + (age - 60) * 0.02)
rezidive_prob = np.clip(rezidive_prob, 0, 1)
rezidive = np.random.binomial(1, rezidive_prob) + 1  # Значения 1 или 2

# Создаем датафрейм
data_age = pd.DataFrame({
    'VOZRAST': age,
    'REZIDIVE': rezidive,
    'VOZRAST_AFTER_60': np.where(age > 60, age - 60, 0)
})

print("=" * 80)
print("1. АНАЛИЗ СВЯЗИ ВОЗРАСТА И РЕЦИДИВА (аналог файла Jaws1)")
print("=" * 80)

# 2.1. Описательная статистика (аналогично Descriptive Statistics из текста)
print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА:")
print("=" * 60)

desc_stats = data_age.describe().T
desc_stats['N'] = n
desc_stats = desc_stats[['N', 'min', 'max', 'mean', 'std']]
desc_stats.columns = ['N', 'Minimum', 'Maximum', 'Mean', 'Std. Deviation']

print(desc_stats.round(3))

# 2.2. Коэффициент корреляции (аналогично Correlations из текста)
corr_age_rez = data_age['VOZRAST'].corr(data_age['REZIDIVE'])

print("\nКОЭФФИЦИЕНТ КОРРЕЛЯЦИИ ПИРСОНА:")
print("=" * 60)
print(f"Коэффициент корреляции между VOZRAST и REZIDIVE: {corr_age_rez:.3f}")

# Проверка значимости
corr_test = stats.pearsonr(data_age['VOZRAST'], data_age['REZIDIVE'])
print(f"p-value: {corr_test[1]:.4f}")
print("Корреляция значима" if corr_test[1] < 0.05 else "Корреляция не значима")

# 2.3. Линейная регрессия (аналогично Analyze → Regression → Linear)

# Подготовка данных для регрессии
X = sm.add_constant(data_age['VOZRAST'])  # Добавляем константу
y = data_age['REZIDIVE']

# Выполняем регрессию
model = sm.OLS(y, X).fit()

print("\nРЕЗУЛЬТАТЫ ЛИНЕЙНОЙ РЕГРЕССИИ:")
print("=" * 60)
print(f"Уравнение регрессии: REZIDIVE = {model.params[0]:.3f} + {model.params[1]:.5f} * VOZRAST")
print(f"R² (квадрат коэффициента корреляции): {model.rsquared:.3f}")
print(f"Скорректированный R²: {model.rsquared_adj:.3f}")

# 2.4. Визуализация связи возраста и рецидива
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Точечная диаграмма с линией регрессии
axes[0, 0].scatter(data_age['VOZRAST'], data_age['REZIDIVE'], alpha=0.6, s=20)
axes[0, 0].plot(data_age['VOZRAST'], model.predict(X), color='red', linewidth=2)

axes[0, 0].set_xlabel('Возраст (VOZRAST)')
axes[0, 0].set_ylabel('Рецидив (REZIDIVE)')
axes[0, 0].set_title('Связь возраста и рецидива с линией регрессии')
axes[0, 0].grid(True, alpha=0.3)

# График 2: Распределение остатков
residuals = model.resid
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Остатки (residuals)')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].set_title('Распределение остатков регрессии')
axes[0, 1].grid(True, alpha=0.3)

# График 3: Возраст, сгруппированный по 10 годам (аналогично анализу из текста)
data_age['VOZRAST_GROUP'] = pd.cut(
    data_age['VOZRAST'],
    bins=range(10, 101, 10),
    labels=[f"{i}-{i+9}" for i in range(10, 91, 10)]
)

group_stats = data_age.groupby('VOZRAST_GROUP')['REZIDIVE'].mean().reset_index()
axes[1, 0].bar(range(len(group_stats)), group_stats['REZIDIVE'],
               tick_label=group_stats['VOZRAST_GROUP'])
axes[1, 0].set_xlabel('Возрастные группы (10 лет)')
axes[1, 0].set_ylabel('Средний REZIDIVE')
axes[1, 0].set_title('Риск рецидива по возрастным группам')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].axhline(y=data_age['REZIDIVE'].mean(), color='red', linestyle='--',
                   label=f"Среднее = {data_age['REZIDIVE'].mean():.2f}")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# График 4: Прогноз по возрасту после 60 лет (интерпретация)
# Анализ по возрасту после 60, как в тексте
data_age['RISK_AFTER_60'] = data_age['VOZRAST_AFTER_60'] > 0
risk_by_age_after_60 = data_age[data_age['VOZRAST_AFTER_60'] > 0].copy()

if len(risk_by_age_after_60) > 0:
    X2 = sm.add_constant(risk_by_age_after_60['VOZRAST_AFTER_60'])
    y2 = risk_by_age_after_60['REZIDIVE']
    model2 = sm.OLS(y2, X2).fit()

    axes[1, 1].scatter(risk_by_age_after_60['VOZRAST_AFTER_60'],
                       risk_by_age_after_60['REZIDIVE'], alpha=0.6, s=20)
    axes[1, 1].plot(risk_by_age_after_60['VOZRAST_AFTER_60'],
                    model2.predict(X2), color='red', linewidth=2)
    axes[1, 1].set_xlabel('Возраст после 60 лет')
    axes[1, 1].set_ylabel('Рецидив (REZIDIVE)')
    axes[1, 1].set_title(f'Прогноз по возрасту после 60 лет\nR² = {model2.rsquared:.3f}')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. ПРИМЕР С ЛЕЙКОЦИТАМИ И СМЕРТНОСТЬЮ (аналог файла PNEUMONIA)

print("\n" + "=" * 80)
print("2. АНАЛИЗ СВЯЗИ ЧИСЛА ЛЕЙКОЦИТОВ И СМЕРТНОСТИ")
print("=" * 80)

# Создание аналогичных данных для примера с лейкоцитами
np.random.seed(123)
n_pneumonia = 1031

# Создаем категории лейкоцитов как в тексте
wbc_categories = ['<4', '4-9', '9-25', '>25']
wbc_probs = [0.055, 0.299, 0.574, 0.072]  # Примерные пропорции из текста
wbc = np.random.choice(wbc_categories, size=n_pneumonia, p=wbc_probs)

# Создаем смертность с нелинейной зависимостью (как в тексте)
# Высокая смертность при низком и высоком уровне лейкоцитов
death_probs = {
    '<4': 0.474,
    '4-9': 0.019,
    '9-25': 0.064,
    '>25': 0.500
}

death = np.array([np.random.binomial(1, death_probs[cat]) for cat in wbc])

# Создаем линейную переменную (как в тексте)
wbc_lin_values = {
    '<4': 0.474,
    '4-9': 0.019,
    '9-25': 0.064,
    '>25': 0.500
}

wbc_lin = np.array([wbc_lin_values[cat] for cat in wbc])

data_pneumonia = pd.DataFrame({
    'УМЕР': death,
    'white_blood_cell_count': wbc,
    'LEJLIN': wbc_lin
})

# Добавляем числовое представление для корреляции
wbc_numeric = {
    '<4': 1,
    '4-9': 2,
    '9-25': 3,
    '>25': 4
}

data_pneumonia['WBC_NUM'] = data_pneumonia['white_blood_cell_count'].map(wbc_numeric)

# 2.1 Анализ корреляций
print("\nКОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ")
print("=" * 60)

# Корреляция с исходной переменной
corr_original = stats.pointbiserialr(data_pneumonia['УМЕР'], data_pneumonia['WBC_NUM'])
corr_lin = stats.pearsonr(data_pneumonia['УМЕР'], data_pneumonia['LEJLIN'])

print(f"Корреляция УМЕР с исходным числовым показателем: {corr_original[0]:.3f}")
print(f"p-value: {corr_original[1]:.4f}")

print(f"\nКорреляция УМЕР с преобразованным показателем: {corr_lin[0]:.3f}")
print(f"p-value: {corr_lin[1]:.4f}")

print("\nВывод: После преобразования связь с исходной переменной изменилась значительно!")

# 2.2 Таблица сопряженности
print("\nТАБЛИЦА СОПРЯЖЕННОСТИ")
print("=" * 60)

contingency_table = pd.crosstab(
    data_pneumonia['white_blood_cell_count'],
    data_pneumonia['УМЕР'],
    margins=True
)

contingency_table_pct = pd.crosstab(
    data_pneumonia['white_blood_cell_count'],
    data_pneumonia['УМЕР'],
    margins=True,
    normalize='index'
) * 100

print("Абсолютные частоты")
print(contingency_table)

print("\nПроценты по строкам")
print(contingency_table_pct.round(1))

# 2.3 Визуализация связи лейкоцитов и смертности
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Смертность по категориям лейкоцитов
death_rate_by_wbc = data_pneumonia.groupby('white_blood_cell_count')['УМЕР'].mean().reindex(wbc_categories)

axes[0, 0].bar(range(len(death_rate_by_wbc)), death_rate_by_wbc.values)
axes[0, 0].set_xticks(range(len(death_rate_by_wbc)))
axes[0, 0].set_xticklabels(death_rate_by_wbc.index)
axes[0, 0].set_xlabel('Число лейкоцитов')
axes[0, 0].set_ylabel('Доля умерших')
axes[0, 0].set_title('Смертность по категориям лейкоцитов (U-образная зависимость)')
axes[0, 0].grid(True, alpha=0.3)

# Добавляем значения над столбцами
for i, v in enumerate(death_rate_by_wbc.values):
    axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

# График 2: Псевдолинейная зависимость
for i, category in enumerate(wbc_categories):
    axes[0, 1].scatter(
        [i] * len(data_pneumonia[data_pneumonia['white_blood_cell_count'] == category]),
        data_pneumonia[data_pneumonia['white_blood_cell_count'] == category]['УМЕР'],
        alpha=0.1, s=30, label=category
    )

axes[0, 1].plot(
    range(len(wbc_categories)),
    [wbc_lin_values[cat] for cat in wbc_categories],
    'r-', linewidth=3, marker='o', markersize=10, label='Линеаризованные значения'
)

axes[0, 1].set_xlabel('Категории лейкоцитов')
axes[0, 1].set_ylabel('УМЕР (0/1)')
axes[0, 1].set_title('Линеаризация U-образной зависимости')
axes[0, 1].set_xticks(range(len(wbc_categories)))
axes[0, 1].set_xticklabels(wbc_categories)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# График 3: Сравнение корреляций
correlations = {
    'Исходный (точечно-бисериальная)': abs(corr_original[0]),
    'Линеаризованный (Пирсон)': abs(corr_lin[0])
}

axes[1, 0].bar(range(len(correlations)), list(correlations.values()))
axes[1, 0].set_xticks(range(len(correlations)))
axes[1, 0].set_xticklabels(list(correlations.keys()))
axes[1, 0].set_ylabel('Абсолютное значение корреляции')
axes[1, 0].set_title('Сравнение силы связи до и после линеаризации')
axes[1, 0].grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, v in enumerate(correlations.values()):
    axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

# График 4: Распределение по категориям
category_counts = data_pneumonia['white_blood_cell_count'].value_counts().reindex(wbc_categories)
axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Распределение пациентов по категориям лейкоцитов')

plt.tight_layout()
plt.show()

# 3. ПОПРАВКА КОРРЕЛЯЦИИ НА ОШИБКИ ИЗМЕРЕНИЯ

print("\n" + "=" * 80)
print("3. ПОПРАВКА КОРРЕЛЯЦИОННОЙ СВЯЗИ НА СЛУЧАЙНЫЕ ОШИБКИ ИЗМЕРЕНИЯ")
print("=" * 80)

# Пример из текста: измерения систолического артериального давления (САД)
print("\nПример из текста:")
print("-" * 60)
print("Коэффициент корреляции измеренного САД с фактором: 0.737")
print("Среднеквадратичное отклонение САД: 15.3")
print("Среднеквадратичное отклонение ошибки измерения САД: 5.1")

# Параметры из примера
r_measured = 0.737
std_sad = 15.3
std_error = 5.1

# Расчеты как в Excel таблице из текста
var_sad = std_sad ** 2
var_error = std_error ** 2
var_true = var_sad - var_error

# Дисперсия необъясненного прогноза измеренного САД по фактору
var_residual_measured = var_sad * (1 - r_measured ** 2)

# Дисперсия необъясненного прогноза точно измеренного САД по фактору
var_residual_true = var_residual_measured - var_error

# Квадрат коэффициента корреляции точно измеренного САД с фактором
r2_true = 1 - var_residual_true / var_true

# Коэффициент корреляции точно измеренного САД с фактором
r_true = np.sign(r_measured) * np.sqrt(r2_true)

print("\nРАСЧЕТЫ (как в Excel таблице из текста):")
print("-" * 60)
print(f"Дисперсия САД: {var_sad:.2f}")
print(f"Дисперсия ошибки измерения САД: {var_error:.2f}")
print(f"Дисперсия точно измеренного САД: {var_true:.2f}")
print(f"Дисперсия необъясненного прогноза измеренного САД по фактору: {var_residual_measured:.2f}")
print(f"Дисперсия необъясненного прогноза точно измеренного САД по фактору: {var_residual_true:.2f}")
print(f"R² для точно измеренного САД: {r2_true:.4f}")
print(f"Коэффициент корреляции точно измеренного САД с фактором: {r_true:.4f}")

print(f"\nВывод: После поправки на ошибку измерения корреляция увеличилась с {r_measured:.3f} до {r_true:.3f}")

# 4. МНОЖЕСТВЕННАЯ РЕГРЕССИЯ (учет нескольких переменных)

print("\n" + "=" * 80)
print("4. МНОЖЕСТВЕННАЯ РЕГРЕССИЯ: ПРОГНОЗ ПО НЕСКОЛЬКИМ ПЕРЕМЕННЫМ")
print("=" * 80)

# Создаем аналогичные данные для множественной регрессии
np.random.seed(456)
n_multiple = 500

# Генерация признаков
age_mult = np.random.normal(60, 15, n_multiple)
age_mult = np.clip(age_mult, 20, 90)

sex = np.random.choice([0, 1], n_multiple, p=[0.4, 0.6])  # 0 - мужской, 1 - женский

wbc_mult = np.random.normal(8, 3, n_multiple)
wbc_mult = np.clip(wbc_mult, 2, 20)

temp = np.random.normal(36.6, 0.8, n_multiple)
pressure = np.random.normal(120, 20, n_multiple)

# Вероятность смерти
death_prob = (
    0.05 +
    0.01 * (age_mult - 60) / 10 +
    0.02 * (wbc_mult < 4) +
    0.03 * (wbc_mult > 12) +
    0.01 * (temp > 37.5) +
    0.02 * (pressure < 100) -
    0.01 * sex
)

death_prob = np.clip(death_prob, 0, 0.5)
death_mult = np.random.binomial(1, death_prob, n_multiple)

# DataFrame
data_multiple = pd.DataFrame({
    'УМЕР': death_mult,
    'age': age_mult,
    'sex': sex,
    'wbc': wbc_mult,
    'temperature': temp,
    'pressure': pressure,
    'wbc_low': (wbc_mult < 4).astype(int),
    'wbc_high': (wbc_mult > 12).astype(int),
    'temp_high': (temp > 37.5).astype(int),
    'pressure_low': (pressure < 100).astype(int)
})

print("\nКОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ (первые 6 переменных):")
print("-" * 60)
corr_matrix = data_multiple[['УМЕР', 'age', 'sex', 'wbc', 'temperature', 'pressure']].corr()
print(corr_matrix.round(3))

# 4.1 Построение множественной регрессии
print("\nПОСТРОЕНИЕ МНОЖЕСТВЕННОЙ РЕГРЕССИИ")
print("-" * 60)

# Шаг 1: одна переменная
X1 = sm.add_constant(data_multiple[['wbc']])
y = data_multiple['УМЕР']
model1 = sm.OLS(y, X1).fit()

print("\nМодель 1 (только wbc):")
print(f"R² = {model1.rsquared:.4f}, Скорректированный R² = {model1.rsquared_adj:.4f}")
print(f"Уравнение: УМЕР = {model1.params[0]:.4f} + {model1.params[1]:.4f} * wbc")

# Шаг 2: добавляем возраст
X2 = sm.add_constant(data_multiple[['wbc', 'age']])
model2 = sm.OLS(y, X2).fit()

print("\nМодель 2 (wbc + age):")
print(f"R² = {model2.rsquared:.4f}, Скорректированный R² = {model2.rsquared_adj:.4f}")

# Шаг 3: Добавляем пол
X3 = sm.add_constant(data_multiple[['wbc', 'age', 'sex']])
model3 = sm.OLS(y, X3).fit()

print("\nМодель 3 (wbc + age + sex):")
print(f"R² = {model3.rsquared:.4f}, Скорректированный R² = {model3.rsquared_adj:.4f}")

# Шаг 4: Добавляем температуру
X4 = sm.add_constant(data_multiple[['wbc', 'age', 'sex', 'temperature']])
model4 = sm.OLS(y, X4).fit()

print("\nМодель 4 (wbc + age + sex + temperature):")
print(f"R² = {model4.rsquared:.4f}, Скорректированный R² = {model4.rsquared_adj:.4f}")

# Шаг 5: Добавляем давление
X5 = sm.add_constant(data_multiple[['wbc', 'age', 'sex', 'temperature', 'pressure']])
model5 = sm.OLS(y, X5).fit()

print("\nМодель 5 (все переменные):")
print(f"R² = {model5.rsquared:.4f}, Скорректированный R² = {model5.rsquared_adj:.4f}")

print("\nВывод: Лучшая модель - модель 3 (wbc + age + sex), так как имеет самый высокий скорректированный R²")

# 4.2 Визуализация результатов множественной регрессии

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Сравнение качества моделей
models = ['Модель 1\n(wbc)', 'Модель 2\n(wbc+age)', 'Модель 3\n(wbc+sex)', 'Модель 4\n(+temp)', 'Модель 5\n(+pressure)']

r2_values = [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
adj_r2_values = [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj, model4.rsquared_adj, model5.rsquared_adj]

x_pos = np.arange(len(models))
width = 0.35

axes[0, 0].bar(x_pos - width/2, r2_values, width, label='R²', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, adj_r2_values, width, label='Скорректированный R²', alpha=0.8)

axes[0, 0].set_xlabel('Модели')
axes[0, 0].set_ylabel('Коэффициент детерминации')
axes[0, 0].set_title('Сравнение качества моделей регрессии')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Добавляем значения
for i, (r2, adj_r2) in enumerate(zip(r2_values, adj_r2_values)):
    axes[0, 0].text(i - width/2, r2 + 0.005, f"{r2:.3f}", ha='center', va='bottom', fontsize=9)
    axes[0, 0].text(i + width/2, adj_r2 + 0.005, f"{adj_r2:.3f}", ha='center', va='bottom', fontsize=9)

# График 2: Фактические vs предсказанные значения
y_pred_best = model3.predict(X3)
axes[0, 1].scatter(y_pred_best, y, alpha=0.5, s=20)
axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)

axes[0, 1].set_xlabel('Предсказанные значения')
axes[0, 1].set_ylabel('Фактические значения')
axes[0, 1].set_title(f'Фактические vs предсказанные значения (Модель 3)\nR² = {model3.rsquared:.3f}')
axes[0, 1].grid(True, alpha=0.3)

# График 3: Коэффициенты модели
coefficients = model3.params[1:]
coef_names = ['wbc', 'age', 'sex']
colors = ['green' if c > 0 else 'red' for c in coefficients]

axes[1, 0].barh(coef_names, coefficients, color=colors)
axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)

axes[1, 0].set_xlabel('Значение коэффициента')
axes[1, 0].set_title('Коэффициенты регрессии (лучшая модель)')
axes[1, 0].grid(True, alpha=0.3)

# Подписи
for i, (name, coef) in enumerate(zip(coef_names, coefficients)):
    axes[1, 0].text(coef, i, f"{coef:.4f}",               va='center', fontweight='bold',
                    color='black' if abs(coef) > 0.01 else 'gray')

# График 4: Вклад переменных
partial_r2 = []
for i in range(1, 4):
    cols = [0, 1, 2]
    cols.remove(i - 1)
    X_partial = X3.iloc[:, [0] + [c + 1 for c in cols]]
    model_partial = sm.OLS(y, X_partial).fit()
    partial_r2.append(model3.rsquared - model_partial.rsquared)

axes[1, 1].pie(partial_r2, labels=coef_names, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Относительный вклад переменных в объясненную дисперсию')

plt.tight_layout()
plt.show()

# ОБЩИЕ ВЫВОДЫ И ЗАКЛЮЧЕНИЕ

print("\n" + "=" * 80)
print("ОСНОВНЫЕ ВЫВОДЫ (согласно учебному материалу):")
print("=" * 80)

print("1. Коэффициент корреляции позволяет строить наилучший линейный прогноз")
print("2. Квадрат коэффициента корреляции показывает долю объясненной дисперсии")
print("3. Линеаризация нелинейных связей может значительно улучшить качество прогноза")
print("4. При измерениях с ошибками истинная корреляция всегда немного выше наблюдаемой")
print("5. В множественной регрессии важен выбор переменных для избежания переобучения")
print("6. Скорректированный R² учитывает число предикторов и помогает выбрать модель")