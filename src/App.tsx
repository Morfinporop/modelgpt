import { useState, useEffect, useRef } from 'react';
import { getAI, MoSeekAI } from './ai/simpleAI';

interface Message {
  id: string;
  type: 'user' | 'ai';
  text?: string;
  image?: string;
  timestamp: Date;
}

function App() {
  const [ai, setAi] = useState<MoSeekAI | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedModel] = useState('MoSeek IMG v1');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    initAI();
    return () => {
      ai?.dispose();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const initAI = async () => {
    try {
      const aiInstance = await getAI();
      setAi(aiInstance);
      setIsReady(true);
      
      // Приветственное сообщение
      setMessages([{
        id: '0',
        type: 'ai',
        text: 'Привет! Я MoSeek IMG v1 - нейросеть для работы с изображениями и текстом.\n\nЯ могу:\n• Генерировать изображения по описанию\n• Анализировать загруженные картинки\n• Отвечать на вопросы\n\nЧем могу помочь?',
        timestamp: new Date()
      }]);
    } catch (error) {
      console.error('Ошибка инициализации AI:', error);
      setIsReady(true);
    }
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if ((!inputText.trim() && !selectedImage) || isProcessing || !ai) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      text: inputText || undefined,
      image: selectedImage || undefined,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);

    const currentInput = inputText;
    const currentImage = selectedImage;
    setInputText('');
    setSelectedImage(null);

    try {
      let aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        timestamp: new Date()
      };

      // Если есть изображение - анализируем его
      if (currentImage) {
        const img = new Image();
        img.onload = async () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            if (ctx && ai) {
              ctx.drawImage(img, 0, 0);
              const imageData = ctx.getImageData(0, 0, img.width, img.height);
              
              // Анализируем изображение
              const analysis = await ai.analyzeImage(imageData);
              
              // Генерируем ответ
              const textResponse = currentInput 
                ? await ai.generateResponse(currentInput, true)
                : '';
              
              aiResponse.text = `${analysis.description}\n\n${textResponse}`.trim();
              setMessages(prev => [...prev, aiResponse]);
            }
          } catch (error) {
            console.error('Ошибка анализа изображения:', error);
            aiResponse.text = 'Произошла ошибка при анализе изображения';
            setMessages(prev => [...prev, aiResponse]);
          } finally {
            setIsProcessing(false);
          }
        };
        img.src = currentImage;
      } 
      // Если запрос на генерацию изображения
      else if (/генерир|создай|нарисуй|сделай|покажи|нарисуй|generate|create|draw|make|show/i.test(currentInput)) {
        try {
          // Генерируем ответ
          const textResponse = await ai.generateResponse(currentInput, false);
          aiResponse.text = textResponse;
          setMessages(prev => [...prev, aiResponse]);
          
          // Генерируем изображение
          setTimeout(async () => {
            try {
              const generatedImage = await ai.generateImage(currentInput);
              
              const imageMessage: Message = {
                id: (Date.now() + 2).toString(),
                type: 'ai',
                image: generatedImage,
                text: 'Вот что получилось!',
                timestamp: new Date()
              };
              
              setMessages(prev => [...prev, imageMessage]);
            } catch (error) {
              console.error('Ошибка генерации:', error);
            } finally {
              setIsProcessing(false);
            }
          }, 500);
        } catch (error) {
          console.error('Ошибка:', error);
          aiResponse.text = 'Произошла ошибка при генерации';
          setMessages(prev => [...prev, aiResponse]);
          setIsProcessing(false);
        }
      } 
      // Обычный текстовый запрос
      else {
        try {
          const textResponse = await ai.generateResponse(currentInput, false);
          aiResponse.text = textResponse;
          setMessages(prev => [...prev, aiResponse]);
          setIsProcessing(false);
        } catch (error) {
          console.error('Ошибка:', error);
          aiResponse.text = 'Произошла ошибка при обработке запроса';
          setMessages(prev => [...prev, aiResponse]);
          setIsProcessing(false);
        }
      }
    } catch (error) {
      console.error('Ошибка:', error);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        text: 'Произошла ошибка при обработке запроса',
        timestamp: new Date()
      }]);
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#2c1810] flex flex-col">
      {/* Header */}
      <div className="bg-[#3d2416] border-b border-[#4d3020] p-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img 
              src="https://raw.githubusercontent.com/tensorflow/tfjs/master/tfjs-core/docs/images/logo.png" 
              alt="Neural" 
              className="w-6 h-6"
            />
            <div className="text-lg font-semibold text-[#d4a574]">MoSeek Neural Network</div>
          </div>
          <div className="flex items-center gap-2">
            <div className="bg-[#4d3020] text-[#d4a574] px-3 py-1 rounded text-sm border border-[#6d4830]">
              {selectedModel}
            </div>
            {!isReady && (
              <div className="text-xs text-[#a67c52]">Загрузка...</div>
            )}
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map(msg => (
            <div
              key={msg.id}
              className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-lg rounded-lg p-3 ${
                  msg.type === 'user'
                    ? 'bg-[#4d3020] text-[#d4a574]'
                    : 'bg-[#3d2416] text-[#c9a676]'
                }`}
              >
                {msg.image && (
                  <img
                    src={msg.image}
                    alt="Message"
                    className="rounded mb-2 max-w-full"
                  />
                )}
                {msg.text && <div className="text-sm whitespace-pre-wrap">{msg.text}</div>}
              </div>
            </div>
          ))}
          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-[#3d2416] text-[#a67c52] rounded-lg p-3">
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-[#a67c52] rounded-full animate-pulse"></div>
                  <span>Обработка запроса...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-[#3d2416] border-t border-[#4d3020] p-4">
        <div className="max-w-4xl mx-auto">
          {selectedImage && (
            <div className="mb-3 relative inline-block">
              <img
                src={selectedImage}
                alt="Preview"
                className="max-h-24 rounded border border-[#6d4830]"
              />
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute -top-2 -right-2 bg-[#6d4830] text-[#d4a574] rounded-full w-5 h-5 flex items-center justify-center text-xs hover:bg-[#8d5830]"
              >
                ✕
              </button>
            </div>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-[#4d3020] text-[#d4a574] px-3 py-2 rounded hover:bg-[#5d3828] transition-colors text-sm"
              disabled={!isReady || isProcessing}
            >
              <img 
                src="https://cdn-icons-png.flaticon.com/512/54/54719.png" 
                alt="Upload" 
                className="w-5 h-5 inline-block opacity-70"
              />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              className="hidden"
            />
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="Введите сообщение..."
              className="flex-1 px-4 py-2 rounded bg-[#4d3020] text-[#d4a574] placeholder-[#8b6f47] border border-[#6d4830] focus:outline-none focus:border-[#8d5830] text-sm"
              disabled={!isReady || isProcessing}
            />
            <button
              onClick={handleSubmit}
              disabled={!isReady || isProcessing || (!inputText.trim() && !selectedImage)}
              className="bg-[#6d4830] hover:bg-[#8d5830] disabled:bg-[#4d3020] disabled:opacity-50 text-[#d4a574] px-6 py-2 rounded transition-colors text-sm"
            >
              Отправить
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
