# Лабораторная работа 5 

Цель лабораторной работы: Обучить нейронную сеть с использованием техники обучения Fine Tuning

## Обучения нейронной сети EfficientNet-B0 с использованием Transfer Learning, техники аугментации данных и политики темпа обучения с оптимальными параметрами.



![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr1](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_categorical_accuracy%20_horizontal.svg)


Функция потерь
![gr2](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_loss_horizontal.svg)

## Использование техники обучения FineTuning

Использовалось два варианта темпа обучения: 1e-7, 1e-8.

```python
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
```
![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/fine.jpg)

### 1e-7
Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_categorical_accuracy_fine_tune(-7).svg)

Функция потерь

![gr4](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_loss_fine_tune(-7).svg)

### 1e-8


Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_categorical_accuracy_fine_tune(-8).svg)

Функция потерь

![gr4](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_loss_fine_tune(-8).svg)

### сравнение


Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/lab5/graphs/epoch_categorical_accuracy_fine_tune.svg)



# 3.Анализ результатов
При использовании техники обучения FineTuning удалось улучшить результаты с 55.3% до 55.7% достигнутые при lr=1e-7 на 6 эпохе.
При использовании техники обучения FineTuning у нас больше обучаемых параметров чем при Transfer Learning и поэтому мне потребовалось больше памяти при обучении сети 
