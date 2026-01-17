# Sentiment Analysis Project

A sentiment analysis toolkit supporting classical ML models, deep learning (MLP), and BERT-based transformers with optional LoRA fine-tuning.

## 🚀 Features

- **Data Preprocessing Pipeline**: Text cleaning, stopword removal, missing value handling
- **Classical ML Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, Decision Tree
- **Deep Learning**: MLP classifier on BERT embeddings
- **Transformer Models**: BERT fine-tuning with optional LoRA for efficient training
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **Rich Visualizations**: EDA plots, training curves, model comparisons, t-SNE embeddings

## 📁 Project Structure

```
sentiment-analysis/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── pipeline.py              # End-to-end pipelines
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Text preprocessing utilities
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── loader.py            # Data loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py         # Sklearn-based models
│   │   └── deep_learning.py     # PyTorch models (MLP, BERT)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loops and utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Evaluation metrics
│   └── visualization/
│       ├── __init__.py
│       └── plots.py             # Plotting utilities
├── notebooks/
│   └── sentiment_analysis.ipynb # Main analysis notebook
├── checkpoints/                 # Model checkpoints
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=1.9.0
transformers>=4.20.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
peft>=0.4.0  # For LoRA
kagglehub>=0.1.0  # For dataset download
```

## 🚀 Quick Start

### Using the Pipeline API

```python
from src.pipeline import SentimentAnalysisPipeline
from src.config import Config

# Initialize pipeline
config = Config()
pipeline = SentimentAnalysisPipeline(config)

# Load data
pipeline.load_data('path/to/train.csv', 'path/to/test.csv')

# Run EDA
pipeline.run_eda()

# Train models
pipeline.train_tfidf_models()      # TF-IDF based models
pipeline.train_embedding_models()  # BERT embedding based models
pipeline.train_mlp(epochs=20)      # MLP on embeddings
pipeline.train_bert(epochs=10)     # Fine-tune BERT

# Compare all models
pipeline.compare_models()

# Get best model
best_name, best_accuracy = pipeline.get_best_model()
print(f"Best model: {best_name} with {best_accuracy:.2%} accuracy")
```

### Using Individual Components

```python
# Text Preprocessing
from src.data.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess("I really don't like this! http://example.com")

# Model Training
from src.models.classical import TFIDFPipeline

pipeline = TFIDFPipeline(classifier_type='logistic_regression')
pipeline.fit(train_texts, train_labels)
predictions = pipeline.predict(test_texts)

# Visualization
from src.visualization.plots import Visualizer

viz = Visualizer(['negative', 'neutral', 'positive'])
viz.plot_training_history(history)
viz.plot_model_comparison(results)
```

## 📊 Model Performance

| Model                        | Accuracy |
| ---------------------------- | -------- |
| TF-IDF + Logistic Regression | ~70%     |
| TF-IDF + Linear SVM          | ~69%     |
| BERT Embeddings + LR         | ~65%     |
| MLP on BERT Embeddings       | ~63%     |
| BERT Fine-tuned              | ~77%     |
| BERT + LoRA                  | ~75%     |

## 🔧 Configuration

Customize the pipeline through the `Config` class:

```python
from src.config import Config, TrainingConfig, ModelConfig

config = Config()

# Modify training settings
config.training.learning_rate = 3e-5
config.training.num_epochs = 15
config.training.batch_size_train = 32

# Modify model settings
config.model.max_length = 256
config.model.pretrained_model = "bert-large-uncased"
```

## 📚 Notebook Walkthrough

The main notebook (`sentiment_analysis.ipynb`) covers:

1. **Data Loading**: Download and load sentiment dataset
2. **Preprocessing**: Clean text, handle missing values
3. **EDA**: Visualize sentiment distributions and word frequencies
4. **Classical Models**: Train and compare TF-IDF based models
5. **BERT Embeddings**: Generate embeddings and train classifiers
6. **Deep Learning**: Train MLP on embeddings
7. **BERT Fine-tuning**: Full fine-tuning and LoRA
8. **Comparison**: Compare all model performances

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- HuggingFace Transformers library
- Kaggle for the sentiment dataset
- BERT paper: "BERT: Pre-training of Deep Bidirectional Transformers"
