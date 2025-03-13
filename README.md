# Train Embeddings MOdel with Pytorch Lightning

Базовый шаблон для обучения эмбеддинг модели на примере face embeddings

## Инструкция по использованию:

1. `make build`

   Соберет образ и установит необходимые зависимости

1. `make run`

   Запустит контейнер с присоединенными папками - `./data, ./checkpoints`.

1. `make train`

   Запустит обучение модели, используя конфигурации из `./configs/traib.yaml`

1. `make convert`

   Сконвертирует полученный чекпионт в **onnx** формат.
   Пример: `make convert MODEL_CHECKPOINT=last.ckpt`, чекпоинт ищется по умолчанию в `./checkpoints`

1. `make inference`

   Запустит инференс на указанной папке с изображениями.
   Ожидаемая структура папки:

   ```
   data/
   ├── person1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── person2/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   ```
