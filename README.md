# House Prices - Advanced Regression Techniques
![image](https://github.com/user-attachments/assets/7c9cbc50-fd99-404d-bdfd-39d1e55bb1a6)

## Описание проекта  
Цель проекта — предсказать конечную цену продажи жилых домов на основе набора характеристик недвижимости. Задача решается как задача регрессии.  
Проект выполнен в рамках практики по машинному обучению и основан на соревновании [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Используемые технологии  
- Python, Pandas, NumPy  
- Визуализация: Matplotlib, Seaborn  
- Машинное обучение: Blending
- Работа с пропущенными значениями и категориальными признаками  
- Feature engineering и отбор признаков

## Этапы работы  
- Исследовательский анализ данных (EDA)  
- Заполнение и обработка пропущенных значений  
- Преобразование категориальных признаков (Label Encoding, One-Hot Encoding)  
- Feature engineering  
- Сравнение нескольких моделей (Blending)
![image](https://github.com/user-attachments/assets/fad02009-ba77-4e7a-a98e-093ef49b7cbc)

## Результаты   
- Score 0.13111


sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
![image](https://github.com/user-attachments/assets/ba0ee36a-4f67-43cf-b3d4-84de0b88a085)



import scipy.stats as stats

y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)
![image](https://github.com/user-attachments/assets/bddc1286-ec6a-46fc-af5f-9c9c3d46a53b)
![image](https://github.com/user-attachments/assets/41e6628c-d7af-4605-8a49-2a63e1ab1b4f)
![image](https://github.com/user-attachments/assets/a7bab9fa-f0b1-4bad-9e37-6a303aec4b40)

