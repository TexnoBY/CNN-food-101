# Лабораторная работа 4 

## Цель 
Исследовать влияние различных техник аугментации данных на процесс обучения нейронной
сети на примере решения задачи классификации Food-101 с использованием техники обучения
Transfer Learning

## Задачи

С использованием, техники обучения Transfer Learning  и оптимальной политики изменения
темпа обучения tf = 0.0001, определенной в ходе выполнения лабораторной #3,
обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений
imagenet) для решения задачи классификации изображений Food-101 с
использованием следующих техник аугментации данных:
+ a. Случайное горизонтальное и вертикальное отображение
+ b. Использование случайной части изображения
+ c. Поворот на случайный угол

## Случайное горизонтальное и вертикальное отображение

### Структура 
mode = ["horizontal", "vertical", "horizontal_and_vertical"]
```python
def build_model(mode):
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  aug_data = tf.keras.layers.experimental.preprocessing.RandomFlip(mode=mode)(inputs)
  x = tf.keras.applications.EfficientNetB0(include_top=False,
                                           weights='imagenet',
                                           input_tensor=aug_data)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
Лучшим Методом оказался "horizontal"
### Графики

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/flip/flip.jpg)

Метрика качества

![gr1](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/flip/epoch_categorical_accuracy_flip.svg)


Функция потерь

![gr2](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/flip/epoch_loss_flip.svg)


## Использование случайной части изображения

### Структура 

```python
def build_model(mode):
  inputs = tf.keras.Input(shape=(250, 250, 3))
  aug_data = tf.keras.layers.experimental.preprocessing.RandomCrop(224, 224)(inputs)
  x = tf.keras.applications.EfficientNetB0(include_top=False,
                                           weights='imagenet',
                                           input_tensor=aug_data)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Графики

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/crop/crop.jpg)

Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/crop/epoch_categorical_accuracy_crop.svg)


Функция потерь

![gr4](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/crop/epoch_loss_crop.svg)

## Поворот на случайный угол

### Структура 
mode = [0.1, 0.3, 0.5, 0.7, 0.9]
method = ['wrap, 'reflect, 'constant']
```python
def build_model(mode):
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  aug_data = tf.keras.layers.experimental.preprocessing.RandomRotation(mode, fill_mode='constant', fill_value=255)(inputs)
  x = tf.keras.applications.EfficientNetB0(include_top=False,
                                           weights='imagenet',
                                           input_tensor=aug_data)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
Лучшей комбинацией оказались angle=0.1, method = 'reflect'
### Графики wrap

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/rotate/wrap/wrap.jpg)

Метрика качества

![gr5](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/rotate/wrap/epoch_categorical_accuracy_wrap.svg)

### Графики reflect

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/rotate/reflect/reflect.jpg)

Метрика качества

![gr5](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/rotate/reflect/epoch_categorical_accuracy_reflect.svg)

### Графики constant

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/rotate/constant/constant.jpg)

Метрика качества

![gr5](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/rotate/constant/epoch_categorical_accuracy_constant.svg)




## Использование нескольких техник

### Структура 

```python
def build_model(mode):
  inputs = tf.keras.Input(shape=(250, 250, 3))
  aug_data = tf.keras.layers.experimental.preprocessing.RandomCrop(224, 224)(inputs)
  aug_data = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal")(aug_data)
  aug_data = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='reflect')(aug_data)
  x = tf.keras.applications.EfficientNetB0(include_top=False,
                                           weights='imagenet',
                                           input_tensor=aug_data)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Графики

![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr7](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/all/epoch_categorical_accuracy%20(2).svg)


Функция потерь

![gr8](https://github.com/TexnoBY/CNN-food-101/blob/lab4/graphs/all/epoch_loss%20(1).svg)



## Вывод


