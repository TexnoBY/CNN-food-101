# Лабораторная работа 1 

В данной лабораторной работе я работал с датасетом Food Images (Food-101)
в котором 101 класс еды вместе с подклассамы(общий объем 101 тысяча картинок).
Целью работы было решить задачу классификации

## 1)Структура
Слой сверетки с 8-ю фильтрами и размером ядра свертки 3х3

```
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
```
Слой пулинга позволяет уменьшить дискретизацию данных посредством выбора максимального значения в окне 
```
x = tf.keras.layers.MaxPool2D()(x)
```
Flatten приводит матрицу признаков к одномерному вектору 
```
x = tf.keras.layers.Flatten()(x)
```
Полностью связанный слой с 20-ю выходами(NUM_CLASSES=20) и функцией активации softmax, которая приводит вероятностную оценку
```
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## 2)Графики 
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr1](https://github.com/TexnoBY/CNN-food-101/blob/master/graphics/epoch_categorical_accuracy%20_1.svg)


Функция потерь
![gr2](https://github.com/TexnoBY/CNN-food-101/blob/master/graphics/epoch_loss%20_1.svg)

# 2.Создание и обучение сверточной нейронной сети произвольной архитектуры с количеством сверточных слоев >3
## 1)Структура
```
 x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
 x = tf.keras.layers.MaxPool2D()(x)
 x = tf.keras.layers.Flatten()(x)
```
Увеличил глубину сети 
```
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
```
## 2)Графики
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/master/graphics/epoch_categorical_accuracy_2.svg)

Функция потерь

![gr4](https://github.com/TexnoBY/CNN-food-101/blob/master/graphics/epoch_loss_2.svg)

# 3.Анализ результатов
Судя по графикам, приведенным выше, после моих модификаций ошибка увеличилась а метрика качества ухудшилась.
Произвольное добавление слоёв привело к более долгому обучению.
