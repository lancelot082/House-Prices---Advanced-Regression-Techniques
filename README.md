# House Prices - Advanced Regression Techniques


## Описание проекта  
Цель проекта — предсказать конечную цену продажи жилых домов на основе набора характеристик недвижимости. Задача решается как задача регрессии.  
Проект выполнен в рамках практики по машинному обучению и основан на соревновании [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Конечно! Вот структурированное и лаконичное описание второго проекта для **README.md**, без эмодзи и без излишнего технического перегруза:

---

## Используемые данные

- `train.csv` — тренировочный набор с признаками и целевой переменной `SalePrice`
- `test.csv` — тестовый набор для предсказания
- `sample_submission.csv` — шаблон для отправки результатов

## Основные этапы пайплайна

### 1. Объединение данных и первичная очистка

- Объединены `train` и `test` в единый датафрейм `df_all` для единообразной обработки
- Удалены признаки с константными и уникальными значениями
- Заполнены пропущенные значения (по моде, медиане, нулями и категориальными заглушками)

### 2. Инженерия признаков

- Созданы дополнительные признаки:
  - `TotalSF`, `Total_sqr_footage`, `Total_Bathrooms`, `Total_porch_sf` и др.
  - Бинарные признаки: наличие бассейна, гаража, подвала, камина и второго этажа
- Переведены категориальные признаки с числовым смыслом в строки
- Категориальные признаки закодированы через **иерархическое (mean) кодирование** и **One-Hot Encoding**
![image](https://github.com/user-attachments/assets/2bfd5b9c-6f46-4ab7-ac72-5570f03c451b)

```python
df_all['Total_Bathrooms'] = (
    df_all['FullBath'] + 0.5 * df_all['HalfBath'] +
    df_all['BsmtFullBath'] + 0.5 * df_all['BsmtHalfBath']
)
```

### 3. Трансформация целевой переменной

- Применено `log1p` преобразование к `SalePrice` для устранения асимметрии
- Проверена пригодность различных распределений (JohnsonSU, Normal, LogNormal)

### 4. Обработка асимметрии признаков

- Рассчитан коэффициент асимметрии для числовых признаков
- К числам с сильной асимметрией применено **Box-Cox преобразование**
![image](https://github.com/user-attachments/assets/637ea2a4-7064-411b-8884-eda474c56cd5)
![image](https://github.com/user-attachments/assets/7df4b8fb-7227-41d0-b5a4-5d777f22f3bf)
![image](https://github.com/user-attachments/assets/e867f397-8693-4414-b93d-b0a2d1365361)

```python
for i in skew_index:
    df_all[i] = boxcox1p(df_all[i], 0)
```

### 5. Нормализация

- Все числовые признаки нормализованы через `StandardScaler`

### 6. Моделирование

- Используются модели:
  - Ridge, Lasso, ElasticNet
  - SVR, GradientBoosting, XGBoost, LightGBM
  - Стекинг (`StackingCVRegressor`) на основе вышеуказанных моделей

- Каждая модель проходит **10-кратную кросс-валидацию**, метрика — `RMSLE`

```python
def cv_rmse(model):
    return np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
```

### 7. Ансамблирование (Blending)

Финальное предсказание — взвешенное среднее от нескольких моделей и стекинга:

```python
def blend_models_predict(X):
    return (0.1 * elastic + 0.1 * ridge + ... + 0.3 * stack_gen.predict(X))
```

### 8. Финальное предсказание

- Предсказанные значения `SalePrice` на тестовом наборе восстанавливаются из логарифма
- Результаты сохраняются в `submission.csv` для отправки на Kaggle

## Метрика

- Используемая метрика: **Root Mean Squared Logarithmic Error (RMSLE)**
- На тренировочных данных достигается стабильный результат с высокой точностью
