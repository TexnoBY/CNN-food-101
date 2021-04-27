# Лабораторная работа 3
#Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Food-101 с использованием техники обучения Transfer Learning

## Learning rate

#### Графики
Использовал фиксированный темп обучения 0.01, 0.001, 0.0001 <br/>
![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/ADAM.jpg)

Метрика качества

![gr1](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/epoch_categorical_accuracy_ADAM.svg)


Функция потерь

![gr2](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/epoch_loss_ADAM.svg)



## Cosine Decay и Cosine Decay with Restarts

### Cosine Decay

Из-за долгого обучения сети и ограниченного времени было проведено обучения
на фиксированном темпе обучения lr = 0.0001, менял лишь decay_steps = [1000, 100, 10]

#### Графики

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/cos_decay.jpg)

Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/epoch_categorical_accuracy_cos_decay.svg)


Функция потерь

![gr4](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/epoch_loss_cos_decay.svg)

### Cosine Decay with Restarts

Тут тоже проводил обучение на фиксированном темпе обучения lr = 0.0001, менял лишь <,>br/>
decay_steps = [1000, 100, 10]

#### Графики

![legend](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/cos_decay_restarts.jpg)

Метрика качества

![gr3](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/epoch_categorical_accuracy_cos_decay_restarts.svg)


Функция потерь

![gr4](https://github.com/TexnoBY/CNN-food-101/blob/lab3/graphics/lab3/epoch_loss_cos_decay_restarts.svg)
## Вывод

При использовании фиксированного темпа обучения
наиболее оптимальным оказался 0.0001 с наибольшей 
точность на валидации. Графики косинусного затухания и косинусного затухания 
с перезапусками оказались схожими на значениях initial_learning_rate = 0.001, decay_steps = 1000,
показали более низкую точность на валидационных данных и быстро переобучились