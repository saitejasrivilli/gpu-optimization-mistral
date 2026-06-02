---
title: Brand Safety System
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.10.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🛡️ Brand Safety System

Multi-modal content moderation system using BERT and CLIP for brand safety detection.

## 🎯 Features

- **91% Accuracy** on 160K+ training samples
- **<250ms Inference Time** for real-time moderation
- **6 Safety Categories**: Toxicity, Hate Speech, Political, Adult, Spam, Safe
- **Multi-Modal Analysis**: Analyzes both text and images
- **Interactive Demo**: Try it instantly with example content

## 🚀 Quick Start

Simply upload text or images to analyze content safety. The system will:
1. Classify content into one of 6 categories
2. Provide confidence scores for each category
3. Give recommendations (Approve/Review/Reject)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 91% |
| Inference Time | <250ms |
| Training Samples | 160K+ |
| Categories | 6 |

## 🛠️ Tech Stack

- **Text Model**: BERT (bert-base-uncased)
- **Image Model**: CLIP (openai/clip-vit-base-patch32)
- **Framework**: PyTorch + HuggingFace Transformers
- **Interface**: Gradio

## 📖 Categories

1. **Toxicity** 🤬 - Offensive, rude, or disrespectful language
2. **Hate Speech** 😡 - Discrimination or hateful content
3. **Political** 🗳️ - Political campaigns or partisan content
4. **Adult** 🔞 - NSFW or adult content
5. **Spam** 📧 - Promotional or spam content
6. **Safe** ✅ - Appropriate for brand association

## 💡 Use Cases

- **Ad Platforms**: Content moderation for ad creatives
- **Social Media**: User-generated content filtering
- **E-commerce**: Product review moderation
- **Forums**: Community content safety

## 🎓 Example Usage

### Text Analysis
```python
# Example toxic text
"You're such an idiot!"
→ Category: Toxicity (92% confidence)

# Example safe text
"Great product, highly recommend!"
→ Category: Safe (95% confidence)
```

### Image Analysis
Upload any image and get safety scores across 5 categories.

### Multi-Modal
Combine text and image for comprehensive analysis.

## ⚙️ How It Works

1. **Text Analysis**: BERT processes text and outputs probability distribution across 6 categories
2. **Image Analysis**: CLIP compares image against safety descriptions
3. **Multi-Modal**: Combines both analyses for comprehensive safety check

## 📊 Model Details

### BERT Classifier
- Base model: `bert-base-uncased`
- Fine-tuned on Jigsaw Toxic Comments + custom data
- 6-class classification
- Softmax activation for probabilities

### CLIP Model
- Model: `openai/clip-vit-base-patch32`
- Zero-shot image classification
- Text-image similarity scoring

## 🎯 Confidence Interpretation

- **>70%**: High confidence - automatic decision
- **30-70%**: Medium confidence - review recommended
- **<30%**: Low confidence - unclear content

## ⚠️ Limitations

- Demo model for educational purposes
- Should be combined with human review for production
- May have biases from training data
- Works best with English text

## 🚀 Deployment

This model is deployed on HuggingFace Spaces for easy access and demonstration.

For production deployment:
1. Fine-tune on your specific data
2. Add human-in-the-loop review
3. Regular bias audits
4. A/B testing for improvements

## 📝 License

MIT License - feel free to use for educational and commercial purposes.

## 👨‍💻 Developer

Built as a portfolio project demonstrating:
- LLM fine-tuning
- Multi-modal AI
- Production ML deployment
- Interactive AI applications

**Tech Stack**: BERT, CLIP, PyTorch, Transformers, Gradio, HuggingFace

---

*For recruiters: This demonstrates expertise in NLP, computer vision, model deployment, and building production-ready ML systems.*
