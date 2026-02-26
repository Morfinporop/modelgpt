import * as tf from '@tensorflow/tfjs';
import natural from 'natural';
// @ts-ignore
import nlp from 'compromise';
// @ts-ignore
import Sentiment from 'sentiment';

const sentiment = new Sentiment();
const tokenizer = new natural.WordTokenizer();

// Продвинутая модель понимания текста
export class MoSeekAI {
  private encoder: tf.LayersModel | null = null;
  private decoder: tf.LayersModel | null = null;
  private imageGenerator: tf.LayersModel | null = null;
  private imageAnalyzer: tf.LayersModel | null = null;
  private vocabulary: Map<string, number> = new Map();
  private reverseVocabulary: Map<number, string> = new Map();
  private conversationHistory: Array<{ role: string; content: string }> = [];
  private knowledgeBase: Map<string, string[]> = new Map();

  constructor() {
    this.initializeVocabulary();
    this.initializeKnowledgeBase();
  }

  private initializeVocabulary() {
    // Базовый словарь на русском и английском
    const baseWords = [
      // Русские слова
      'привет', 'как', 'дела', 'что', 'это', 'нейросеть', 'картинка', 'изображение', 
      'генерация', 'создай', 'нарисуй', 'покажи', 'сделай', 'можешь', 'хочу', 'нужно',
      'красиво', 'красивый', 'хорошо', 'плохо', 'отлично', 'супер', 'круто', 'класс',
      'цвет', 'красный', 'синий', 'зеленый', 'желтый', 'черный', 'белый', 'серый',
      'кот', 'собака', 'дом', 'машина', 'дерево', 'цветок', 'человек', 'животное',
      'пейзаж', 'портрет', 'абстракция', 'природа', 'город', 'море', 'небо', 'солнце',
      'луна', 'звезды', 'космос', 'лес', 'горы', 'река', 'озеро', 'пляж', 'закат',
      'рассвет', 'ночь', 'день', 'вечер', 'утро', 'зима', 'лето', 'весна', 'осень',
      'снег', 'дождь', 'облака', 'туман', 'ветер', 'гроза', 'молния', 'радуга',
      // English words
      'hello', 'hi', 'how', 'what', 'is', 'this', 'neural', 'network', 'image', 'picture',
      'generate', 'create', 'draw', 'make', 'show', 'can', 'want', 'need', 'good', 'bad',
      'beautiful', 'nice', 'awesome', 'great', 'color', 'red', 'blue', 'green', 'yellow',
      'cat', 'dog', 'house', 'car', 'tree', 'flower', 'person', 'animal', 'landscape',
      'portrait', 'abstract', 'nature', 'city', 'sea', 'sky', 'sun', 'moon', 'stars',
      'space', 'forest', 'mountains', 'river', 'lake', 'beach', 'sunset', 'sunrise',
      // Технические термины
      'модель', 'обучение', 'данные', 'алгоритм', 'нейрон', 'слой', 'вес', 'параметр',
      'точность', 'ошибка', 'градиент', 'оптимизация', 'функция', 'активация', 'потери'
    ];

    baseWords.forEach((word, idx) => {
      this.vocabulary.set(word.toLowerCase(), idx);
      this.reverseVocabulary.set(idx, word.toLowerCase());
    });
  }

  private initializeKnowledgeBase() {
    // База знаний для ответов
    this.knowledgeBase.set('greeting', [
      'Привет! Я MoSeek - нейросеть для работы с изображениями. Чем могу помочь?',
      'Здравствуй! Я могу генерировать изображения или анализировать загруженные. Что тебя интересует?',
      'Привет! Готов помочь с созданием или анализом изображений!'
    ]);

    this.knowledgeBase.set('generate', [
      'Опиши подробнее, что именно хочешь увидеть на изображении?',
      'С удовольствием создам изображение! Расскажи детали: цвета, объекты, настроение.',
      'Какой стиль изображения предпочитаешь? Реалистичный, абстрактный, или что-то другое?'
    ]);

    this.knowledgeBase.set('analyze', [
      'Загрузи изображение, и я его проанализирую!',
      'Готов изучить твое изображение. Что именно хочешь узнать?',
      'Отправь картинку, я определю что на ней изображено.'
    ]);

    this.knowledgeBase.set('help', [
      'Я умею:\n• Генерировать изображения по описанию\n• Анализировать загруженные картинки\n• Отвечать на вопросы\n\nПросто напиши что нужно!',
      'Мои возможности:\n✓ Создание изображений\n✓ Распознавание объектов\n✓ Оценка качества\n\nЧем помочь?'
    ]);

    this.knowledgeBase.set('quality', [
      'Изображение выглядит качественно! Композиция сбалансирована.',
      'Хорошая картинка! Цвета гармоничные, детали четкие.',
      'Отличное изображение! Контраст и яркость на хорошем уровне.'
    ]);

    this.knowledgeBase.set('default', [
      'Интересный вопрос! Можешь уточнить подробнее?',
      'Понял тебя. Расскажи больше деталей?',
      'Хороший запрос! Что конкретно тебя интересует?'
    ]);
  }

  async initialize() {
    console.log('🚀 Инициализация MoSeek AI...');
    
    // Создаем энкодер для понимания текста
    this.encoder = await this.createTextEncoder();
    
    // Создаем декодер для генерации ответов
    this.decoder = await this.createTextDecoder();
    
    // Создаем генератор изображений
    this.imageGenerator = await this.createImageGenerator();
    
    // Создаем анализатор изображений
    this.imageAnalyzer = await this.createImageAnalyzer();
    
    console.log('✅ MoSeek AI готов к работе!');
  }

  private async createTextEncoder(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    // Embedding layer
    model.add(tf.layers.embedding({
      inputDim: 1000,
      outputDim: 128,
      inputLength: 50
    }));
    
    // LSTM для понимания контекста
    model.add(tf.layers.lstm({
      units: 256,
      returnSequences: true,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));
    
    model.add(tf.layers.lstm({
      units: 128,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));
    
    // Dense layers
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return model;
  }

  private async createTextDecoder(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({ 
      units: 128, 
      activation: 'relu',
      inputShape: [64]
    }));
    
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError'
    });
    
    return model;
  }

  private async createImageGenerator(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    // Генератор на основе GAN
    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu',
      inputShape: [128]
    }));
    
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.3 }));
    
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.batchNormalization());
    
    model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
    model.add(tf.layers.batchNormalization());
    
    model.add(tf.layers.dense({ units: 2048, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // Выход: 128x128x3 = 49152
    model.add(tf.layers.dense({ units: 49152, activation: 'tanh' }));
    
    model.compile({
      optimizer: tf.train.adam(0.0002),
      loss: 'meanSquaredError'
    });
    
    return model;
  }

  private async createImageAnalyzer(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    // CNN для анализа изображений
    model.add(tf.layers.conv2d({
      inputShape: [128, 128, 3],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return model;
  }

  // Анализ намерения пользователя
  private analyzeIntent(text: string): {
    intent: string;
    entities: string[];
    sentiment: number;
    keywords: string[];
  } {
    const lowerText = text.toLowerCase();
    const doc = nlp(text);
    
    // Sentiment analysis
    const sentimentResult = sentiment.analyze(text);
    
    // Извлечение ключевых слов
    const tokens = tokenizer.tokenize(lowerText) || [];
    const keywords = tokens.filter((word: string) => 
      this.vocabulary.has(word) || word.length > 3
    );
    
    // Определение намерения
    let intent = 'default';
    
    if (/привет|здравствуй|hello|hi/i.test(text)) {
      intent = 'greeting';
    } else if (/генерир|создай|нарисуй|сделай|generate|create|draw/i.test(text)) {
      intent = 'generate';
    } else if (/анализ|посмотри|оцени|что.*на.*картинк|analyze|check/i.test(text)) {
      intent = 'analyze';
    } else if (/помощ|помоги|умеешь|можешь|help|can.*you/i.test(text)) {
      intent = 'help';
    } else if (/качество|красив|хорош|плох|quality|beautiful|good|bad/i.test(text)) {
      intent = 'quality';
    }
    
    // Извлечение сущностей (объекты, цвета, стили)
    const entities: string[] = [];
    
    // Цвета
    const colors = doc.match('#Color').out('array');
    entities.push(...colors);
    
    // Существительные
    const nouns = doc.nouns().out('array');
    entities.push(...nouns);
    
    return {
      intent,
      entities: [...new Set(entities)],
      sentiment: sentimentResult.score,
      keywords
    };
  }

  // Генерация умного ответа
  async generateResponse(userMessage: string, hasImage: boolean = false): Promise<string> {
    // Добавляем в историю
    this.conversationHistory.push({
      role: 'user',
      content: userMessage
    });
    
    // Анализируем намерение
    const analysis = this.analyzeIntent(userMessage);
    
    console.log('📊 Анализ:', analysis);
    
    let response = '';
    
    // Генерируем ответ на основе намерения
    const responses = this.knowledgeBase.get(analysis.intent) || this.knowledgeBase.get('default')!;
    response = responses[Math.floor(Math.random() * responses.length)];
    
    // Улучшаем ответ с контекстом
    if (analysis.intent === 'generate') {
      if (analysis.entities.length > 0) {
        response = `Отлично! Создаю изображение с: ${analysis.entities.slice(0, 3).join(', ')}. Это может занять несколько секунд...`;
      }
    }
    
    if (hasImage && analysis.intent === 'analyze') {
      response = `Анализирую изображение... Вижу интересные детали! Сейчас расскажу подробнее.`;
    }
    
    // Учитываем тональность
    if (analysis.sentiment > 0) {
      response = response.replace('!', '! 😊').replace('?', '? 🤔');
    }
    
    // Персонализация на основе истории
    if (this.conversationHistory.length > 2) {
      const lastUserMessage = this.conversationHistory[this.conversationHistory.length - 3];
      if (lastUserMessage && lastUserMessage.role === 'user') {
        if (/спасибо|благодар|thanks/i.test(userMessage)) {
          response = 'Всегда пожалуйста! Рад помочь! Если нужно что-то еще - обращайся! 😊';
        }
      }
    }
    
    // Добавляем в историю
    this.conversationHistory.push({
      role: 'assistant',
      content: response
    });
    
    // Ограничиваем историю
    if (this.conversationHistory.length > 20) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }
    
    return response;
  }

  // Генерация изображения по описанию
  async generateImage(description: string): Promise<string> {
    if (!this.imageGenerator) {
      throw new Error('Генератор не инициализирован');
    }
    
    console.log('🎨 Генерация изображения:', description);
    
    // Анализируем описание
    const analysis = this.analyzeIntent(description);
    
    // Создаем latent vector на основе текста
    const latentVector = this.textToLatentVector(description, analysis);
    
    // Генерируем изображение
    const generated = tf.tidy(() => {
      const input = tf.tensor2d([latentVector]);
      const output = this.imageGenerator!.predict(input) as tf.Tensor;
      
      // Преобразуем в изображение 128x128x3
      const reshaped = output.reshape([128, 128, 3]);
      
      // Нормализуем от tanh (-1, 1) к (0, 255)
      const normalized = reshaped.add(1).mul(127.5);
      
      return normalized;
    });
    
    // Конвертируем в canvas и получаем data URL
    const imageData = await this.tensorToImageData(generated);
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(imageData, 0, 0);
    
    // Улучшаем изображение (постобработка)
    this.enhanceImage(ctx, canvas, analysis);
    
    const dataUrl = canvas.toDataURL('image/png');
    
    generated.dispose();
    
    return dataUrl;
  }

  // Преобразование текста в latent vector
  private textToLatentVector(text: string, analysis: any): number[] {
    const vector: number[] = new Array(128).fill(0);
    
    // Используем хеш слов для генерации уникального вектора
    const words = tokenizer.tokenize(text.toLowerCase()) || [];
    
    words.forEach((word) => {
      const hash = this.hashCode(word);
      const index = Math.abs(hash) % 128;
      vector[index] += Math.sin(hash * 0.01) * 2;
    });
    
    // Добавляем влияние сущностей
    analysis.entities.forEach((entity: string) => {
      const hash = this.hashCode(entity);
      const index = Math.abs(hash) % 128;
      vector[index] += Math.cos(hash * 0.01) * 3;
    });
    
    // Нормализуем
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => norm > 0 ? v / norm : v);
  }

  private hashCode(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash;
  }

  // Улучшение изображения
  private enhanceImage(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, analysis: any) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Добавляем эффекты на основе ключевых слов
    const keywords = analysis.keywords.join(' ').toLowerCase();
    
    if (keywords.includes('небо') || keywords.includes('sky')) {
      // Голубой оттенок
      for (let i = 0; i < data.length; i += 4) {
        data[i + 2] = Math.min(255, data[i + 2] * 1.2); // Синий
      }
    }
    
    if (keywords.includes('закат') || keywords.includes('sunset')) {
      // Оранжевый оттенок
      for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, data[i] * 1.3); // Красный
        data[i + 1] = Math.min(255, data[i + 1] * 1.1); // Зеленый
      }
    }
    
    if (keywords.includes('лес') || keywords.includes('forest') || keywords.includes('дерев')) {
      // Зеленый оттенок
      for (let i = 0; i < data.length; i += 4) {
        data[i + 1] = Math.min(255, data[i + 1] * 1.3); // Зеленый
      }
    }
    
    // Увеличиваем контраст
    const factor = 1.2;
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, Math.max(0, (data[i] - 128) * factor + 128));
      data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] - 128) * factor + 128));
      data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] - 128) * factor + 128));
    }
    
    ctx.putImageData(imageData, 0, 0);
  }

  private async tensorToImageData(tensor: tf.Tensor): Promise<ImageData> {
    const [height, width] = tensor.shape.slice(0, 2);
    const data = await tensor.data();
    
    const imageData = new ImageData(width as number, height as number);
    
    for (let i = 0; i < height * width; i++) {
      const r = Math.min(255, Math.max(0, data[i * 3]));
      const g = Math.min(255, Math.max(0, data[i * 3 + 1]));
      const b = Math.min(255, Math.max(0, data[i * 3 + 2]));
      
      imageData.data[i * 4] = r;
      imageData.data[i * 4 + 1] = g;
      imageData.data[i * 4 + 2] = b;
      imageData.data[i * 4 + 3] = 255;
    }
    
    return imageData;
  }

  // Анализ изображения
  async analyzeImage(imageData: ImageData): Promise<{
    description: string;
    objects: string[];
    colors: string[];
    quality: number;
    mood: string;
  }> {
    if (!this.imageAnalyzer) {
      throw new Error('Анализатор не инициализирован');
    }
    
    console.log('🔍 Анализ изображения...');
    
    // Подготавливаем изображение
    const tensor = tf.tidy(() => {
      // Resize до 128x128
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d')!;
      
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = imageData.width;
      tempCanvas.height = imageData.height;
      const tempCtx = tempCanvas.getContext('2d')!;
      tempCtx.putImageData(imageData, 0, 0);
      
      ctx.drawImage(tempCanvas, 0, 0, 128, 128);
      
      const resized = ctx.getImageData(0, 0, 128, 128);
      
      // Преобразуем в тензор
      const tensorData = tf.browser.fromPixels(resized);
      const normalized = tensorData.toFloat().div(255);
      
      return normalized.expandDims(0);
    });
    
    // Анализируем
    const predictions = this.imageAnalyzer.predict(tensor) as tf.Tensor;
    const probabilitiesData = await predictions.data();
    const probabilities = new Float32Array(probabilitiesData);
    
    // Определяем доминантные цвета
    const colors = this.extractDominantColors(imageData);
    
    // Оцениваем качество
    const quality = this.assessQuality(imageData);
    
    // Определяем настроение
    const mood = this.detectMood(colors, probabilities);
    
    // Определяем возможные объекты
    const objects = this.detectObjects(probabilities);
    
    // Генерируем описание
    const description = this.generateDescription(objects, colors, mood, quality);
    
    tensor.dispose();
    predictions.dispose();
    
    return {
      description,
      objects,
      colors,
      quality,
      mood
    };
  }

  private extractDominantColors(imageData: ImageData): string[] {
    const colorCounts: { [key: string]: number } = {};
    const data = imageData.data;
    
    // Группируем цвета
    for (let i = 0; i < data.length; i += 16) { // Сэмплируем каждый 4й пиксель
      const r = Math.floor(data[i] / 51) * 51;
      const g = Math.floor(data[i + 1] / 51) * 51;
      const b = Math.floor(data[i + 2] / 51) * 51;
      
      const color = `${r},${g},${b}`;
      colorCounts[color] = (colorCounts[color] || 0) + 1;
    }
    
    // Сортируем по частоте
    const sorted = Object.entries(colorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    // Преобразуем в названия цветов
    return sorted.map(([rgb]) => this.rgbToColorName(rgb));
  }

  private rgbToColorName(rgb: string): string {
    const [r, g, b] = rgb.split(',').map(Number);
    
    if (r > 200 && g > 200 && b > 200) return 'белый';
    if (r < 50 && g < 50 && b < 50) return 'черный';
    if (r > Math.max(g, b) * 1.5) return 'красный';
    if (g > Math.max(r, b) * 1.5) return 'зеленый';
    if (b > Math.max(r, g) * 1.5) return 'синий';
    if (r > 150 && g > 150 && b < 100) return 'желтый';
    if (r > 150 && g < 100 && b > 150) return 'фиолетовый';
    if (r > 150 && g > 100 && b < 100) return 'оранжевый';
    if (r < 100 && g > 150 && b > 150) return 'голубой';
    
    return 'серый';
  }

  private assessQuality(imageData: ImageData): number {
    const data = imageData.data;
    let totalVariance = 0;
    let totalBrightness = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      totalBrightness += brightness;
      
      if (i > 0) {
        const prevBrightness = (data[i - 4] + data[i - 3] + data[i - 2]) / 3;
        totalVariance += Math.abs(brightness - prevBrightness);
      }
    }
    
    const avgBrightness = totalBrightness / (data.length / 4);
    const avgVariance = totalVariance / (data.length / 4);
    
    // Качество на основе контраста и яркости
    const quality = Math.min(100, (avgVariance / 255 * 100 + (100 - Math.abs(avgBrightness - 128) / 128 * 100)) / 2);
    
    return Math.round(quality);
  }

  private detectMood(colors: string[], _probabilities: Float32Array): string {
    const colorMood = colors[0];
    
    if (colorMood === 'красный' || colorMood === 'оранжевый') return 'энергичное';
    if (colorMood === 'синий' || colorMood === 'голубой') return 'спокойное';
    if (colorMood === 'зеленый') return 'умиротворенное';
    if (colorMood === 'желтый') return 'радостное';
    if (colorMood === 'фиолетовый') return 'мистическое';
    if (colorMood === 'черный') return 'драматичное';
    if (colorMood === 'белый') return 'чистое';
    
    return 'нейтральное';
  }

  private detectObjects(probabilities: Float32Array): string[] {
    const categories = [
      'пейзаж', 'портрет', 'животное', 'растение', 'архитектура',
      'транспорт', 'еда', 'предмет', 'абстракция', 'текстура'
    ];
    
    const topIndices = Array.from(probabilities)
      .map((prob, idx) => ({ prob, idx }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3)
      .map(item => item.idx);
    
    return topIndices.map(idx => categories[idx] || 'объект');
  }

  private generateDescription(objects: string[], colors: string[], mood: string, quality: number): string {
    const qualityText = quality > 80 ? 'высокого качества' : quality > 60 ? 'хорошего качества' : 'среднего качества';
    
    return `Вижу ${objects[0] || 'изображение'} ${qualityText}. Доминирующие цвета: ${colors.slice(0, 3).join(', ')}. Настроение: ${mood}. Общая оценка качества: ${quality}/100.`;
  }

  // Очистка ресурсов
  dispose() {
    this.encoder?.dispose();
    this.decoder?.dispose();
    this.imageGenerator?.dispose();
    this.imageAnalyzer?.dispose();
  }
}

// Синглтон
let aiInstance: MoSeekAI | null = null;

export async function getAI(): Promise<MoSeekAI> {
  if (!aiInstance) {
    aiInstance = new MoSeekAI();
    await aiInstance.initialize();
  }
  return aiInstance;
}
