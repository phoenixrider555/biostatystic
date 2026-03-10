import numpy as np
import pandas as pd

# Исходные данные (число выздоровевших / общее число пациентов)
# Формат:  [исход_1, общее_1, исход_2, общее_2]
raw = {
    "Урюпинск":      [96, 100, 98, 100],
    "Бобруйск":      [85, 100, 84, 100],
    "Ю.Бутово":      [100, 104, 100, 103],
    "Дом престарелых": [100, 141, 100, 124],
    "ВЧ №13":        [100, 100, 100, 101],
    "студенты 1МГМУ": [96, 100, 100, 101],
    "СОВОКУПНОЕ":    [1000, 1009, 1000, 1010]   # примерные суммированные числа
}

df = pd.DataFrame.from_dict(raw, orient='index',
                            columns=['Success1', 'N1', 'Success2', 'N2'])
df.head()

# Вычисление вспомогательных величин
# Число неуспехов
df['Fail1'] = df['N1'] - df['Success1']
df['Fail2'] = df['N2'] - df['Success2']

# Частоты успеха
df['p1'] = df['Success1'] / df['N1']
df['p2'] = df['Success2'] / df['N2']

# Шансы (odds)
df['odds1'] = df['Success1'] / df['Fail1']
df['odds2'] = df['Success2'] / df['Fail2']

# Оценка RR и OR + доверительные интервалы
from scipy.stats import norm

z = norm.ppf(0.975)          # 1.96 для 95 % CI

# Relative Risk
df['RR']   = df['p1'] / df['p2']
df['logRR']= np.log(df['RR'])
df['se_logRR'] = np.sqrt( (1/df['Success1'] - 1/df['N1']) +
                          (1/df['Success2'] - 1/df['N2']) )
df['RR_low']  = np.exp(df['logRR'] - z*df['se_logRR'])
df['RR_high'] = np.exp(df['logRR'] + z*df['se_logRR'])

# Odds Ratio
df['OR']   = df['odds1'] / df['odds2']
df['logOR']= np.log(df['OR'])
df['se_logOR'] = np.sqrt( 1/df['Success1'] + 1/df['Fail1'] +
                          1/df['Success2'] + 1/df['Fail2'] )
df['OR_low']  = np.exp(df['logOR'] - z*df['se_logOR'])
df['OR_high'] = np.exp(df['logOR'] + z*df['se_logOR'])

# Таблица результатов
cols = ['RR','RR_low','RR_high','OR','OR_low','OR_high']
print(df[cols].round(3))
# Forest plot (график)
import matplotlib.pyplot as plt
import seaborn as sns

def forest_plot(df, effect_col, low_col, high_col, title):
    """Отрисовка forest plot для любого эффекта (RR или OR)."""
    fig, ax = plt.subplots(figsize=(6, len(df)*0.6))
    
    # Позиции по оси Y (от нижнего к верхнему)
    y_pos = np.arange(len(df), 0, -1)

    # Горизонтальные линии (доверительные интервалы)
    ax.hlines(y_pos, df[low_col], df[high_col], color='gray')
    # Точки – точечные оценки
    ax.scatter(df[effect_col], y_pos, color='steelblue', zorder=3)

    # Вертикальная линия в 1 (нулевой эффект)
    ax.axvline(1, color='red', linestyle='--')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.set_xlabel(effect_col)
    ax.set_title(title)
    ax.invert_yaxis()          # чтобы первая строка была сверху
    plt.tight_layout()
    plt.show()

# RR plot
forest_plot(df, 'RR', 'RR_low', 'RR_high',
            'Относительный риск (RR) с 95 % ДИ')
# OR plot
forest_plot(df, 'OR', 'OR_low', 'OR_high',
            'Отношение шансов (OR) с 95 % ДИ')
