# Инструкция по деплою на Railway.com

## Быстрый старт

1. **Загрузите проект на GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MoSeek Neural Network"
   git branch -M main
   git remote add origin https://github.com/Morfinporop/MoSeek.git
   git push -u origin main
   ```

2. **Подключите к Railway**
   - Зайдите на railway.app
   - Нажмите "New Project"
   - Выберите "Deploy from GitHub repo"
   - Выберите репозиторий `Morfinporop/MoSeek`

3. **Railway автоматически**
   - Обнаружит Node.js проект
   - Прочитает настройки из `railway.json`
   - Установит Node.js 20.x (из `.nvmrc`)
   - Выполнит `npm install` и `npm run build`
   - Запустит сервер командой `npm start`

4. **Проверьте настройки**
   - **Networking → Public Networking**: Убедитесь что домен активен
   - **Target Port**: Должен быть 8080
   - **Settings → Deploy**: Branch должен быть `main`

## Настройки Railway (уже сконфигурированы)

Все настройки уже включены в проект:

- ✅ `railway.json` - конфигурация сборки и деплоя
- ✅ `Procfile` - команды запуска
- ✅ `.nvmrc` - версия Node.js 20.x
- ✅ `server.js` - Express сервер на порту 8080
- ✅ `package.json` - скрипты build и start

## Автоматический деплой

После настройки каждый push в `main` ветку будет автоматически:
1. Триггерить новую сборку
2. Запускать `npm run build`
3. Деплоить новую версию
4. Переключать трафик на новую версию

## Проверка работоспособности

После деплоя откройте ваш домен:
- `https://mogpt.up.railway.app`

Должно загрузиться приложение MoSeek Neural Network с:
- Кофейным темным интерфейсом
- Выбором модели "MoSeek IMG v1"
- Полем ввода текста
- Кнопкой загрузки файла

## Troubleshooting

### Ошибка сборки
- Проверьте что Node.js версия 20.x
- Убедитесь что все зависимости в `package.json`
- Проверьте логи сборки в Railway

### Приложение не запускается
- Проверьте что порт 8080 правильно настроен
- Убедитесь что `npm start` работает локально
- Проверьте переменные окружения

### Нейросеть не работает
- TensorFlow.js загружается в браузере, проверьте консоль браузера
- Убедитесь что JavaScript включен
- Проверьте совместимость браузера (используйте Chrome/Firefox/Edge)

## Дополнительные настройки

### Custom Domain
1. В Railway перейдите в Networking
2. Нажмите "Custom Domain"
3. Добавьте ваш домен
4. Настройте DNS записи у вашего регистратора

### Environment Variables
Текущая версия не требует переменных окружения, но вы можете добавить:
- `NODE_ENV=production` (опционально)
- `PORT=8080` (уже настроен в коде)

## Мониторинг

Railway предоставляет:
- Логи в реальном времени
- Метрики использования CPU/RAM
- История деплоев
- Автоматические алерты

Готово! Ваша нейросеть MoSeek теперь работает в продакшене! 🚀
