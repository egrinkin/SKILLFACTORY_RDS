## Прогнозирование цены автомобиля по характеристикам, текстовому описанию и фото
### Цель
Обогатив датасет текстовыми данными из объявлений о продаже, свести все модели в единое решение — Multi-inputs сеть.

### Описание кейса
Имеются характеристики автомобилей, их текстовое описание и фото. Необходимо с максимально возможной точностью предсказывать стоимость этих авто. 

### Выводы
**EDA**
* Данные проверены на наличие дубликатов и выбросов.
* Показано, что исходные числовые признаки распределены ненормально.

**Feature Engineering**
* Извлечены числовые значения из текстовых данных путем парсинга признаков 'engineDisplacement', 'enginePower'.
* Признаки 'modelDate' и 'ProductionDate'преобразованы к возрасту модели и автомобиля соответственно.
* Числовые признаки нормированы.
* Удалены ненужные признаки.
* Заполнены пропуски в значениях как числовых, так и категориальных признаков. 
* Cокращена размерность категориальных признаков: из признака 'name' удалены данные, которые содержатся в других признаках ('enginePower', 'engineDisplacement', 'vehicleTransmission').

**ML**
* Для предотворащения утечки данных при стандартизации признаков применен Pipeline.
* Осуществлен решетчатный поиск лучших параметров модели Сastboost Regressor с кросс-валидацией на 5 фолдах.

**NLP**
* Из исходных текстовых данных исключены стоп-слова.
* Слова в текстовых данных приведены к нормальной форме. 

**Ассамблирование моделей**
* Применен подход, заключающийся в обучении новой модели на метапризнаках, которые предсавтялют собой предсказания ML- и DL-моделей на тренировочных данных. 

Достигнутое значение матрики на [Kaggle](https://www.kaggle.com/c/sf-dst-car-price-prediction-part2/leaderboard) - 12.41810

Nickname - Evgeniy Grinkin

