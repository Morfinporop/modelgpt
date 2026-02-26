import * as tf from '@tensorflow/tfjs';

export class ImageGenerator {
  private generator: tf.LayersModel | null = null;
  private isTraining = false;
  private latentDim = 100;
  private imageSize = 64;

  async initialize() {
    console.log('Initializing MoSeek IMG v1 Generator...');
    this.generator = this.buildGenerator();
    console.log('MoSeek IMG v1 Generator ready');
  }

  private buildGenerator(): tf.LayersModel {
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.dense({
      inputShape: [this.latentDim],
      units: 256,
      useBias: true
    }));
    model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    model.add(tf.layers.batchNormalization());

    // Hidden layers
    model.add(tf.layers.dense({ units: 512 }));
    model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.dense({ units: 1024 }));
    model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.dense({ units: 2048 }));
    model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    model.add(tf.layers.batchNormalization());

    // Output layer - RGB image
    model.add(tf.layers.dense({ 
      units: this.imageSize * this.imageSize * 3,
      activation: 'tanh'
    }));
    
    model.add(tf.layers.reshape({ targetShape: [this.imageSize, this.imageSize, 3] }));

    return model;
  }

  async generateImage(seed?: number): Promise<ImageData> {
    if (!this.generator) {
      throw new Error('Generator not initialized');
    }

    // Generate random latent vector or use seed
    let noise: tf.Tensor;
    if (seed !== undefined) {
      const seedValue = seed;
      noise = tf.randomNormal([1, this.latentDim], 0, 1, 'float32', seedValue);
    } else {
      noise = tf.randomNormal([1, this.latentDim]);
    }

    // Generate image
    const generated = this.generator.predict(noise) as tf.Tensor;
    
    // Convert to ImageData format
    const imageArray = generated.reshape([this.imageSize, this.imageSize, 3]);
    const normalized = imageArray.add(1).div(2).mul(255).clipByValue(0, 255);
    
    const data = await normalized.data();
    const uint8Data = new Uint8ClampedArray(data);
    
    // Cleanup tensors
    noise.dispose();
    generated.dispose();
    imageArray.dispose();
    normalized.dispose();
    
    return new ImageData(uint8Data, this.imageSize, this.imageSize);
  }

  async trainOnData(iterations: number, onProgress?: (iteration: number, loss: number) => void): Promise<void> {
    if (!this.generator) {
      throw new Error('Generator not initialized');
    }

    this.isTraining = true;
    console.log(`Training MoSeek IMG v1 for ${iterations} iterations...`);

    const optimizer = tf.train.adam(0.0002, 0.5);

    for (let i = 0; i < iterations; i++) {
      let lossValue = 0;
      
      await optimizer.minimize(() => {
        // Generate random latent vectors
        const noise = tf.randomNormal([16, this.latentDim]);
        
        // Generate images
        const generated = this.generator!.predict(noise) as tf.Tensor;
        
        // Simple training: push towards diverse, colorful images
        const target = tf.randomUniform([16, this.imageSize, this.imageSize, 3], -1, 1);
        const loss = tf.losses.meanSquaredError(target, generated);
        
        lossValue = (loss as tf.Scalar).dataSync()[0];
        
        return loss as tf.Scalar;
      });

      if (onProgress && i % 10 === 0) {
        onProgress(i, lossValue);
      }

      await tf.nextFrame();
    }

    this.isTraining = false;
    console.log('Training complete');
  }

  getImageSize(): number {
    return this.imageSize;
  }

  isGeneratorTraining(): boolean {
    return this.isTraining;
  }

  dispose() {
    if (this.generator) {
      this.generator.dispose();
    }
  }
}
