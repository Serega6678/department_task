## Отборочное задание на кафедру КЛ Abbyy

Прежде всего, спасибо за интереснейшее задание!

Кратко моя реализация:
1. При помощи предобученного Tiny Ru Bert (работает очень быстро) считаются эмбеддинги каждого сегмента внутри текста. Эта часть не обучается.
2. На основе эмбеддингов модель RNN учится делать предсказания смены автора. Функция ошибки - BCE с весом в 17.15 раз больше на позитивный класс (ради балансировки классов).

Метрики на тест (лучший f1 на валидационной выборке достигается при):
* thr = 0.69
* f1 = 0.51 (0.53)
* precision = 0.94 (0.82)
* recall = 0.35 (0.39)

Также в голову пришла альтернативная идея решения задачи:
так как обычно вывод о смене автора можно делать по первым/последним словам в сегменте, то идея была построить мешок слов по первым + последним словам сегмента и на нем уже учить классификатор (так можно было бы избежать использования Tiny Ru Bert).

Технические требования:
1. Python 3.8
2. Poetry
3. Make

Установка:
```
make bootstrap
```

Разбитие данных на train/val/test:
```
make split_data
```

Предподсчет эмбеддингов Bert на train/val/test:
```
make extract_data
```

Обучение модели:
```
make train
```

Подбор оптимального порога по тестовой выборке:
```
make estimate_thr
```

Предсказание на тесте (**пользователь указать в Makefile тресхолд для классификации**, также в Makefile можно заменить путь до файла с текстами и предсказывать ответы на любой текст):
```
make predict_test
```

Замер метрики:
```
make measure_metric
```

Еще раз спасибо за интересное задание, до встречи на собеседовании 😉