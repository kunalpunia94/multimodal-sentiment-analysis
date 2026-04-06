# Detailed Plan: Audio-Visual Emotion Recognition with Perceiver IO

## 🔹 Paper Summary: RAVER

The paper "Contextual Attention for Robust Audio-Visual Emotion Recognition (RAVER)" introduces a model for emotion recognition that is robust to missing audio or video modalities.

### Architecture:
*   **Audio Encoder**: WavLM, a self-supervised model for speech.
*   **Video Encoder**: MobileNetV2, a lightweight CNN for vision.
*   **Fusion**: A Transformer-based "Context Summarizer" that attends to both modality features.

### Key Contributions:
*   Handles missing modalities through modality dropout during training.
*   Achieves competitive results on CREMA-D and MSP-IMPROV datasets.
*   The fusion mechanism is based on a transformer encoder that processes concatenated features.

## 🔹 Limitations of RAVER

*   **CNN-based Vision**: While MobileNetV2 is efficient, it may not capture complex facial expressions as effectively as modern transformer-based vision models like ViT or Swin Transformer.
*   **Limited Fusion Mechanism**: The context summarizer uses a standard transformer encoder on concatenated features. This may not be the most efficient or effective way to fuse information from two very different modalities.
*   **Audio Model**: While WavLM is powerful, other models like HuBERT have shown superior performance on various speech-related tasks.

## 🔹 Our Improvements

We propose to build a stronger model by incorporating more advanced components:

*   **Swin/ViT for Vision**: We will use a pre-trained Swin Transformer or Vision Transformer (ViT) as the video encoder. These models have demonstrated superior performance on a wide range of computer vision tasks and are better suited to capture the nuances of facial expressions.
*   **HuBERT for Audio**: We will use a pre-trained HuBERT model as the audio encoder. HuBERT has shown excellent performance in learning rich representations from raw audio.
*   **Perceiver IO for Fusion**: We will implement a Perceiver IO model for modality fusion. The Perceiver is a highly efficient and scalable architecture that can handle high-dimensional inputs and learn to fuse them effectively through cross-attention with a small set of latent tokens. This is a more sophisticated approach than simple concatenation and self-attention.

## 🔹 Expected Outcome

By using more powerful encoders and a more advanced fusion mechanism, we expect our model to achieve:

*   **Better F1 Score**: Higher macro and micro F1 scores, indicating better overall performance.
*   **Better Robustness**: The Perceiver IO's attention mechanism is expected to handle missing modalities more gracefully, leading to improved performance when one modality is absent.
*   **State-of-the-Art Results**: The goal is to set a new state-of-the-art on the CREMA-D and MSP-IMPROV datasets for audio-visual emotion recognition.
