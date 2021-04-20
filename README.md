# Лабораторная работа 2
# Решение задачи классификации изображений из набора данных Food-101 с использованием нейронных сетей глубокого обучения и техники обучения Transfer Learning


##EfficientNet
train.py
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
outputs = tf.keras.applications.EfficientNetB0(include_top=True,
                                             weights=None,
                                             classes=NUM_CLASSES,
                                             classifier_activation='softmax',
                                             input_tensor=inputs)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
### Графики
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr1](https://github.com/TexnoBY/CNN-food-101/blob/lab2/graphics/lab2/epoch_categorical_accuracy.svg)


Функция потерь
![gr2](https://github.com/TexnoBY/CNN-food-101/blob/lab2/graphics/lab2/epoch_loss.svg)

##EfficientNet с Transfer learning

train_transfer.py
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
x = tf.keras.applications.EfficientNetB0(include_top=False,
                                       weights='imagenet',
                                       input_tensor=inputs)
x.trainable = False
x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
### Графики
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr1](https://github.com/TexnoBY/CNN-food-101/blob/lab2/graphics/lab2/transfer_epoch_categorical_accuracy.svg)


Функция потерь
![gr2](https://github.com/TexnoBY/CNN-food-101/blob/lab2/graphics/lab2/transfer_epoch_loss.svg)

## Вывод
Первая попытка обучения была неудовлетворительной, максимальная достигнутая точность около 30%.
Во второй раз при использовании Transfer learning произошло значительное улучшение результатов,
но также появилось явное переобучение сети. При использовании Transfer learning сеть обучалась в разы быстрее.
По моим размышлениям в первом случае не хватает данных для лучшего обучения(в imagenet dataset 14млн картинок),
поэтому надо провести аугментацию данных. 