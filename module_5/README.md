## Определение модели авто по фото

### Цель
Построение модели классификации изображений с помощью нейронных сетей.
### Описание кейса

Компания занимающется продажей автомобилей с пробегом. Менеджеры тратят много времени на поиск автомобилей в базе. Система должна выдавать информацию по фотографии. 

### Выводы
Пошагово произведено улучшение базовой модели: 
1. [Выполнен разведочный анализ данных](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-01-eda-xception.ipynb)
2. [Выполнена продвинутая аугментация данных](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-02-augmentation.ipynb)
3. [Проведены эксперименты по варьированию размера изображения](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-03-image-size.ipynb)
4. [Улучшена "голова"](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-04-custom-head.ipynb)
5. [Применены функции callback](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-05-callback.ipynb)
6. [Применены EFFICIENTNETB0](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-06-efficientnetb0.ipynb) и [EFFICIENTNETB7](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-07-efficientnetb7.ipynb) 
7. [Проведены эксперименты по варьированию количества эпох](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-08-epochs-number.ipynb)
8. [Применен fine-tuning](https://github.com/egrinkin/SKILLFACTORY_RDS/blob/main/module_5/car-classification-step-09-fine-tuning.ipynb)


Эти действия позволили последовательно улучшать метрику и достичь ее значения 0.97213 - [Kaggle](https://www.kaggle.com/c/sf-dl-car-classification/leaderboard)

Nickname - Evgeniy Grinkin 