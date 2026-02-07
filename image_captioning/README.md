# Image Captioning with Deep Learning

This project generates natural language captions for images using a combination of convolutional neural networks and recurrent neural networks. The model was trained on the Flickr8k dataset.

## Architecture Overview

The system consists of two main components that work together:

### Feature Extraction (Xception CNN)

Xception is a deep convolutional network pre-trained on ImageNet. We use it as a feature extractor by removing its final classification layer. The model takes a 299x299 pixel image and outputs a 2048-dimensional feature vector that captures the visual content of the image.

### Caption Generation (LSTM)

The caption model is an LSTM-based sequence generator that takes image features and generates text one word at a time.

The image features and LSTM features are merged through simple addition, creating a combined representation that informs the next word prediction.

Deployed: [Test](https://image-caption-ankit.streamlit.app/)
