import * as tf from '@tensorflow/tfjs';

export interface ClassificationResult {
  quality: number;
  colorfulness: number;
  complexity: number;
  aesthetic: number;
}

export class ImageClassifier {
  private classifier: tf.LayersModel | null = null;
  private imageSize = 64;

  async initialize() {
    console.log('Initializing MoSeek IMG v1 Classifier...');
    this.classifier = this.buildClassifier();
    await this.trainClassifier();
    console.log('MoSeek IMG v1 Classifier ready');
  }

  private buildClassifier(): tf.LayersModel {
    const model = tf.sequential();
    
    // Conv layers
    model.add(tf.layers.conv2d({
      inputShape: [this.imageSize, this.imageSize, 3],
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
    
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    
    // Output: 4 scores (quality, colorfulness, complexity, aesthetic)
    model.add(tf.layers.dense({ units: 4, activation: 'sigmoid' }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['accuracy']
    });
    
    return model;
  }

  private async trainClassifier(): Promise<void> {
    if (!this.classifier) return;

    console.log('Training classifier...');

    // Generate synthetic training data
    const batchSize = 32;
    const epochs = 20;

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Create random images
      const images = tf.randomUniform([batchSize, this.imageSize, this.imageSize, 3], 0, 1);
      
      // Create synthetic labels based on image statistics
      const labels = tf.tidy(() => {
        const mean = images.mean([1, 2, 3]);
        const variance = images.sub(mean.reshape([batchSize, 1, 1, 1])).square().mean([1, 2, 3]);
        
        // Calculate std dev manually
        const mean2d = images.mean([1, 2]);
        const diff = images.sub(mean2d.reshape([batchSize, 1, 1, 3]));
        const colorfulness = diff.square().mean([1, 2]).sqrt().mean(1);
        
        const maxValues = images.max([1, 2, 3]);
        
        return tf.stack([
          mean,
          colorfulness,
          variance,
          maxValues
        ], 1);
      });
      
      await this.classifier!.fit(images, labels, {
        epochs: 1,
        verbose: 0
      });
      
      // Cleanup
      images.dispose();
      labels.dispose();

      await tf.nextFrame();
    }

    console.log('Classifier training complete');
  }

  async classifyImage(imageData: ImageData): Promise<ClassificationResult> {
    if (!this.classifier) {
      throw new Error('Classifier not initialized');
    }

    return await tf.tidy(() => {
      // Convert ImageData to tensor
      const tensor = tf.browser.fromPixels(imageData)
        .resizeBilinear([this.imageSize, this.imageSize])
        .toFloat()
        .div(255.0)
        .expandDims(0);
      
      // Get predictions
      const predictions = this.classifier!.predict(tensor) as tf.Tensor;
      const scores = predictions.dataSync();
      
      return {
        quality: Math.round(scores[0] * 100),
        colorfulness: Math.round(scores[1] * 100),
        complexity: Math.round(scores[2] * 100),
        aesthetic: Math.round(scores[3] * 100)
      };
    });
  }

  async analyzeImage(imageData: ImageData): Promise<{
    classification: ClassificationResult;
    statistics: {
      brightness: number;
      contrast: number;
      saturation: number;
    };
  }> {
    const classification = await this.classifyImage(imageData);
    
    // Calculate image statistics
    const statistics = await tf.tidy(() => {
      const tensor = tf.browser.fromPixels(imageData).toFloat().div(255.0);
      
      const brightness = tensor.mean().dataSync()[0];
      const variance = tensor.sub(brightness).square().mean().dataSync()[0];
      const contrast = Math.sqrt(variance);
      
      // Simple saturation calculation
      const r = tensor.slice([0, 0, 0], [-1, -1, 1]);
      const g = tensor.slice([0, 0, 1], [-1, -1, 1]);
      const b = tensor.slice([0, 0, 2], [-1, -1, 1]);
      
      const max = tf.maximum(tf.maximum(r, g), b);
      const min = tf.minimum(tf.minimum(r, g), b);
      const saturation = max.sub(min).mean().dataSync()[0];
      
      return {
        brightness: Math.round(brightness * 100),
        contrast: Math.round(contrast * 100),
        saturation: Math.round(saturation * 100)
      };
    });
    
    return { classification, statistics };
  }

  dispose() {
    if (this.classifier) {
      this.classifier.dispose();
    }
  }
}
