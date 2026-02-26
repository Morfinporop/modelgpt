import * as tf from '@tensorflow/tfjs';

// Простая и легкая AI без тяжелых зависимостей
export class MoSeekAI {
  private encoder: tf.LayersModel | null = null;
  private imageGenerator: tf.LayersModel | null = null;
  private imageAnalyzer: tf.LayersModel | null = null;
  private conversationHistory: Array<{ role: string; content: string }> = [];
  private vocabulary: Map<string, number[]> = new Map();

  constructor() {
    this.initializeVocabulary();
  }

  private initializeVocabulary() {
    // Расширенный словарь для понимания
    const keywords = {
      greeting: ['привет', 'здравствуй', 'hello', 'hi', 'hey', 'добрый'],
      generate: ['генерир', 'создай', 'нарисуй', 'сделай', 'покажи', 'нарисуй', 'generate', 'create', 'draw', 'make', 'show'],
      analyze: ['анализ', 'посмотри', 'оцени', 'что.*на.*картинк', 'analyze', 'check', 'разбери'],
      help: ['помощ', 'помоги', 'умеешь', 'можешь', 'help', 'can'],
      thanks: ['спасибо', 'благодар', 'thanks', 'thank'],
      quality: ['качество', 'красив', 'хорош', 'плох', 'quality', 'beautiful', 'good', 'bad'],
      colors: ['красный', 'синий', 'зеленый', 'желтый', 'черный', 'белый', 'серый', 'фиолетовый', 'оранжевый', 'голубой',
               'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'purple', 'orange', 'cyan'],
      objects: ['кот', 'собака', 'дом', 'машина', 'дерево', 'цветок', 'человек', 'животное', 'cat', 'dog', 'house', 'car', 'tree', 'flower', 'person', 'animal'],
      nature: ['пейзаж', 'природа', 'лес', 'горы', 'река', 'озеро', 'море', 'небо', 'солнце', 'луна', 'звезды', 'космос',
               'landscape', 'nature', 'forest', 'mountains', 'river', 'lake', 'sea', 'sky', 'sun', 'moon', 'stars', 'space'],
      weather: ['закат', 'рассвет', 'ночь', 'день', 'вечер', 'утро', 'зима', 'лето', 'весна', 'осень', 'снег', 'дождь',
                'sunset', 'sunrise', 'night', 'day', 'evening', 'morning', 'winter', 'summer', 'spring', 'autumn', 'snow', 'rain']
    };

    Object.entries(keywords).forEach(([category, words]) => {
      words.forEach(word => {
        this.vocabulary.set(word.toLowerCase(), this.stringToVector(category));
      });
    });
  }

  private stringToVector(str: string): number[] {
    const vector: number[] = new Array(16).fill(0);
    for (let i = 0; i < str.length; i++) {
      const charCode = str.charCodeAt(i);
      vector[i % 16] += charCode;
    }
    return vector.map(v => v / 1000);
  }

  async initialize() {
    console.log('🚀 Инициализация MoSeek AI...');
    
    this.encoder = await this.createTextEncoder();
    this.imageGenerator = await this.createImageGenerator();
    this.imageAnalyzer = await this.createImageAnalyzer();
    
    console.log('✅ MoSeek AI готов!');
  }

  private async createTextEncoder(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    model.add(tf.layers.embedding({
      inputDim: 1000,
      outputDim: 64,
      inputLength: 20
    }));
    
    model.add(tf.layers.lstm({
      units: 128,
      returnSequences: false
    }));
    
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy'
    });
    
    return model;
  }

  private async createImageGenerator(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    // Простой генератор
    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu',
      inputShape: [64]
    }));
    
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2048, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 49152, activation: 'tanh' })); // 128x128x3
    
    model.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError'
    });
    
    return model;
  }

  private async createImageAnalyzer(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
      inputShape: [128, 128, 3],
      filters: 32,
      kernelSize: 3,
      activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return model;
  }

  // Анализ намерения
  private analyzeIntent(text: string): {
    intent: string;
    keywords: string[];
    sentiment: number;
  } {
    const lower = text.toLowerCase();
    let intent = 'chat';
    const keywords: string[] = [];
    let sentiment = 0;
    
    // Проверяем намерения
    if (/привет|здравствуй|hello|hi/i.test(text)) {
      intent = 'greeting';
      sentiment = 1;
    } else if (/генерир|создай|нарисуй|сделай|покажи|generate|create|draw|make|show/i.test(text)) {
      intent = 'generate';
      sentiment = 0.5;
    } else if (/анализ|посмотри|оцени|analyze|check/i.test(text)) {
      intent = 'analyze';
      sentiment = 0;
    } else if (/помощ|помоги|умеешь|можешь|help|can/i.test(text)) {
      intent = 'help';
      sentiment = 0;
    } else if (/спасибо|благодар|thanks/i.test(text)) {
      intent = 'thanks';
      sentiment = 1;
    }
    
    // Извлекаем ключевые слова
    const words = lower.split(/\s+/);
    for (const word of words) {
      for (const [key] of this.vocabulary) {
        if (word.includes(key) || key.includes(word)) {
          keywords.push(key);
        }
      }
    }
    
    return { intent, keywords: [...new Set(keywords)], sentiment };
  }

  // Генерация ответа
  async generateResponse(userMessage: string, hasImage: boolean = false): Promise<string> {
    this.conversationHistory.push({
      role: 'user',
      content: userMessage
    });
    
    const analysis = this.analyzeIntent(userMessage);
    let response = '';
    
    switch (analysis.intent) {
      case 'greeting':
        response = this.getRandomResponse([
          'Привет! Я MoSeek IMG v1 - нейросеть для работы с изображениями. Чем могу помочь?',
          'Здравствуй! Готов помочь с генерацией или анализом изображений!',
          'Привет! Расскажи, что нужно создать или какую картинку проанализировать?'
        ]);
        break;
        
      case 'generate':
        if (analysis.keywords.length > 0) {
          const objects = analysis.keywords.slice(0, 3).join(', ');
          response = `Понял! Генерирую изображение с элементами: ${objects}. Сейчас создам для тебя картинку!`;
        } else {
          response = 'Отлично! Опиши подробнее что хочешь увидеть: цвета, объекты, настроение.';
        }
        break;
        
      case 'analyze':
        if (hasImage) {
          response = 'Анализирую изображение... Сейчас расскажу что на нем!';
        } else {
          response = 'Загрузи изображение, и я его проанализирую!';
        }
        break;
        
      case 'help':
        response = 'Мои возможности:\n\n✓ Генерация изображений по описанию\n✓ Анализ загруженных картинок\n✓ Ответы на вопросы\n\nПросто напиши что нужно создать или загрузи картинку!';
        break;
        
      case 'thanks':
        response = 'Пожалуйста! Всегда рад помочь! Если нужно еще что-то - обращайся!';
        break;
        
      default:
        if (hasImage) {
          response = 'Интересная картинка! Сейчас проанализирую её подробнее.';
        } else {
          response = `Понял тебя! ${analysis.keywords.length > 0 ? `Вижу ты упомянул: ${analysis.keywords.slice(0, 3).join(', ')}.` : ''}\n\nЯ могу:\n• Сгенерировать изображение (скажи "создай...")\n• Проанализировать картинку (загрузи изображение)\n\nЧто тебя интересует?`;
        }
    }
    
    this.conversationHistory.push({
      role: 'assistant',
      content: response
    });
    
    if (this.conversationHistory.length > 20) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }
    
    return response;
  }

  private getRandomResponse(responses: string[]): string {
    return responses[Math.floor(Math.random() * responses.length)];
  }

  // Генерация изображения
  async generateImage(description: string): Promise<string> {
    if (!this.imageGenerator) {
      throw new Error('Генератор не инициализирован');
    }
    
    console.log('🎨 Генерация изображения:', description);
    
    const analysis = this.analyzeIntent(description);
    const latentVector = this.textToLatentVector(description, analysis);
    
    const generated = tf.tidy(() => {
      const input = tf.tensor2d([latentVector]);
      const output = this.imageGenerator!.predict(input) as tf.Tensor;
      const reshaped = output.reshape([128, 128, 3]);
      const normalized = reshaped.add(1).mul(127.5);
      return normalized;
    });
    
    const imageData = await this.tensorToImageData(generated);
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(imageData, 0, 0);
    
    // Улучшаем изображение
    this.enhanceImage(ctx, canvas, analysis);
    
    const dataUrl = canvas.toDataURL('image/png');
    generated.dispose();
    
    return dataUrl;
  }

  private textToLatentVector(text: string, analysis: any): number[] {
    const vector: number[] = new Array(64).fill(0);
    
    // Используем хеш текста
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      vector[i % 64] += Math.sin(charCode * 0.1) * 2;
    }
    
    // Добавляем влияние ключевых слов
    analysis.keywords.forEach((keyword: string, idx: number) => {
      const hash = this.hashString(keyword);
      vector[idx % 64] += Math.cos(hash * 0.01) * 1.5;
    });
    
    // Нормализуем
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => norm > 0 ? v / norm * 2 : v);
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return hash;
  }

  private enhanceImage(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, analysis: any) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    const keywords = analysis.keywords.join(' ').toLowerCase();
    
    // Применяем цветовые фильтры на основе ключевых слов
    if (keywords.includes('небо') || keywords.includes('sky')) {
      for (let i = 0; i < data.length; i += 4) {
        data[i + 2] = Math.min(255, data[i + 2] * 1.3);
      }
    }
    
    if (keywords.includes('закат') || keywords.includes('sunset')) {
      for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, data[i] * 1.4);
        data[i + 1] = Math.min(255, data[i + 1] * 1.2);
      }
    }
    
    if (keywords.includes('лес') || keywords.includes('forest') || keywords.includes('дерев')) {
      for (let i = 0; i < data.length; i += 4) {
        data[i + 1] = Math.min(255, data[i + 1] * 1.4);
      }
    }
    
    if (keywords.includes('ночь') || keywords.includes('night')) {
      for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.max(0, data[i] * 0.6);
        data[i + 1] = Math.max(0, data[i + 1] * 0.6);
        data[i + 2] = Math.min(255, data[i + 2] * 1.2);
      }
    }
    
    // Увеличиваем контраст
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, Math.max(0, (data[i] - 128) * 1.3 + 128));
      data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] - 128) * 1.3 + 128));
      data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] - 128) * 1.3 + 128));
    }
    
    ctx.putImageData(imageData, 0, 0);
  }

  private async tensorToImageData(tensor: tf.Tensor): Promise<ImageData> {
    const [height, width] = tensor.shape.slice(0, 2);
    const data = await tensor.data();
    
    const imageData = new ImageData(width as number, height as number);
    
    for (let i = 0; i < height * width; i++) {
      imageData.data[i * 4] = Math.min(255, Math.max(0, data[i * 3]));
      imageData.data[i * 4 + 1] = Math.min(255, Math.max(0, data[i * 3 + 1]));
      imageData.data[i * 4 + 2] = Math.min(255, Math.max(0, data[i * 3 + 2]));
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
      
      const tensorData = tf.browser.fromPixels(resized);
      return tensorData.toFloat().div(255).expandDims(0);
    });
    
    // Анализируем
    const predictions = this.imageAnalyzer.predict(tensor) as tf.Tensor;
    const probabilities = await predictions.data();
    
    // Анализируем цвета
    const colors = this.extractDominantColors(imageData);
    
    // Оцениваем качество
    const quality = this.assessQuality(imageData);
    
    // Определяем настроение
    const mood = this.detectMood(colors);
    
    // Определяем объекты
    const objects = this.detectObjects(Array.from(probabilities));
    
    // Генерируем описание
    const description = this.generateDescription(objects, colors, mood, quality);
    
    tensor.dispose();
    predictions.dispose();
    
    return { description, objects, colors, quality, mood };
  }

  private extractDominantColors(imageData: ImageData): string[] {
    const colorCounts: { [key: string]: number } = {};
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 16) {
      const r = Math.floor(data[i] / 51) * 51;
      const g = Math.floor(data[i + 1] / 51) * 51;
      const b = Math.floor(data[i + 2] / 51) * 51;
      
      const color = `${r},${g},${b}`;
      colorCounts[color] = (colorCounts[color] || 0) + 1;
    }
    
    const sorted = Object.entries(colorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
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
    
    return 'серый';
  }

  private assessQuality(imageData: ImageData): number {
    const data = imageData.data;
    let totalContrast = 0;
    let totalBrightness = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      totalBrightness += brightness;
      
      if (i > 0) {
        const prevBrightness = (data[i - 4] + data[i - 3] + data[i - 2]) / 3;
        totalContrast += Math.abs(brightness - prevBrightness);
      }
    }
    
    const avgBrightness = totalBrightness / (data.length / 4);
    const avgContrast = totalContrast / (data.length / 4);
    
    const quality = Math.min(100, (avgContrast / 255 * 100 + (100 - Math.abs(avgBrightness - 128) / 128 * 100)) / 2);
    
    return Math.round(quality);
  }

  private detectMood(colors: string[]): string {
    const mainColor = colors[0];
    
    if (mainColor === 'красный' || mainColor === 'оранжевый') return 'энергичное';
    if (mainColor === 'синий' || mainColor === 'голубой') return 'спокойное';
    if (mainColor === 'зеленый') return 'умиротворенное';
    if (mainColor === 'желтый') return 'радостное';
    if (mainColor === 'фиолетовый') return 'мистическое';
    if (mainColor === 'черный') return 'драматичное';
    if (mainColor === 'белый') return 'чистое';
    
    return 'нейтральное';
  }

  private detectObjects(probabilities: number[]): string[] {
    const categories = [
      'пейзаж', 'портрет', 'животное', 'растение', 'архитектура',
      'транспорт', 'еда', 'предмет', 'абстракция', 'текстура'
    ];
    
    const topIndices = probabilities
      .map((prob, idx) => ({ prob, idx }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3)
      .map(item => item.idx);
    
    return topIndices.map(idx => categories[idx] || 'объект');
  }

  private generateDescription(objects: string[], colors: string[], mood: string, quality: number): string {
    const qualityText = quality > 80 ? 'высокого качества' : quality > 60 ? 'хорошего качества' : 'среднего качества';
    
    return `Анализ изображения:\n\nТип: ${objects[0] || 'изображение'}\nКачество: ${qualityText} (${quality}/100)\nЦвета: ${colors.slice(0, 3).join(', ')}\nНастроение: ${mood}`;
  }

  dispose() {
    this.encoder?.dispose();
    this.imageGenerator?.dispose();
    this.imageAnalyzer?.dispose();
  }
}

let aiInstance: MoSeekAI | null = null;

export async function getAI(): Promise<MoSeekAI> {
  if (!aiInstance) {
    aiInstance = new MoSeekAI();
    await aiInstance.initialize();
  }
  return aiInstance;
}
