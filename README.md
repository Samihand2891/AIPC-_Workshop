# Insurance Policy Analysis Engine - AI-Powered Document Intelligence

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Insurance Policy Analysis Engine** is an end-to-end AI-powered system designed to automatically extract, structure, and analyze information from insurance policy documents. Built using state-of-the-art Natural Language Processing (NLP) techniques, this system transforms static, unstructured policy documents into queryable, actionable data assets.

### Key Capabilities
- 📄 **Intelligent Document Processing (IDP)** - Handles both digital-native and scanned PDF documents
- 🤖 **Domain-Specific NER** - Extracts insurance-specific entities (policy numbers, coverage limits, dates, parties)
- 📊 **Layout-Aware Analysis** - Processes complex form layouts (ACORD forms, declarations pages)
- 🔗 **Relation Extraction** - Builds knowledge graphs by identifying relationships between entities
- 🌐 **Interactive Web Interface** - User-friendly Streamlit application for document upload and analysis

### Business Value
- ⚡ **Automated Claims Processing** - Reduce claim adjudication time from days to minutes
- 🎯 **Accurate Risk Assessment** - Extract policy terms and conditions for underwriting decisions
- 🔍 **Compliance Verification** - Ensure policies meet regulatory requirements
- 💰 **Cost Reduction** - Minimize manual data entry and human error

---

## ✨ Features

### 1. Multi-Format Document Ingestion
- **Digital-native PDFs**: Fast text extraction using PyMuPDF
- **Scanned Documents**: OCR processing with automatic image enhancement
- **Hybrid Documents**: Intelligent routing between extraction methods
- **Table Extraction**: Specialized handling of premium schedules and coverage tables

### 2. Advanced NLP Analysis
- **Named Entity Recognition (NER)**
  - Insured Party identification
  - Policy Number extraction
  - Coverage Limits and Deductibles
  - Effective/Expiration Dates
  - Insurer and Broker information

- **Relation Extraction**
  - Policy ownership relationships
  - Coverage associations
  - Temporal relationships
  - Multi-party connections

### 3. Interactive Web Application
- Drag-and-drop file upload
- Real-time processing status
- Visual entity highlighting
- Structured JSON output
- Downloadable results

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                    (Streamlit Web App)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Pipeline Orchestrator                       │
│                   (PolicyAnalyzer)                           │
└─────┬────────────────┬────────────────┬─────────────────────┘
      │                │                │
┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼──────┐
│ Ingestion │   │    NER    │   │  Relation  │
│  Gateway  │   │  Service  │   │ Extraction │
└─────┬─────┘   └─────┬─────┘   └─────┬──────┘
      │               │               │
┌─────▼───────────────▼───────────────▼──────┐
│          Data Persistence Layer             │
│     (Structured JSON / Database)            │
└─────────────────────────────────────────────┘
```

### Architecture Principles
- **Event-Driven Design**: Asynchronous processing for scalability
- **Microservices-Inspired**: Independent, loosely-coupled components
- **API-First**: RESTful interface for enterprise integration
- **Cloud-Ready**: Containerizable for AWS/Azure/GCP deployment

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for large documents)
- **Storage**: 5GB free space for models and temporary files
- **GPU** (Optional): CUDA-compatible GPU for faster model inference

### Required Software
- Python 3.8+
- Tesseract OCR (for scanned document processing)
- Git (for cloning the repository)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/insurance-policy-analyzer.git
cd insurance-policy-analyzer
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR

#### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to System PATH:
   ```cmd
   setx /M PATH "%PATH%;C:\Program Files\Tesseract-OCR"
   ```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Step 5: Download Pre-trained Models

#### Option A: Download from Releases
```bash
# Download and extract models from GitHub releases
wget https://github.com/your-username/insurance-policy-analyzer/releases/download/v1.0/models.zip
unzip models.zip -d models/
```

#### Option B: Train Your Own Models
See [Model Training](#model-training) section below.

#### Option C: Use Base Models (No Fine-tuning)
Update `config.yaml`:
```yaml
model_paths:
  ner: "nlpaueb/legal-bert-base-uncased"
  layout: null  # Disable layout analysis
  relations: null  # Disable relation extraction
```

### Step 6: Verify Installation

```bash
python test_import.py
```

Expected output:
```
✓ DocumentProcessor imported successfully!
✓ Processor initialized with threshold: 100
✓ All dependencies properly configured!
```

---

## 📁 Project Structure

```
insurance-policy-analyzer/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # System configuration
├── .gitignore                         # Git ignore rules
│
├── app.py                            # Streamlit web application
├── pipeline.py                       # Main analysis orchestrator
├── ingestion.py                      # PDF processing module
├── nlp_models.py                     # Model loading utilities
│
├── scripts/                          # Training and utility scripts
│   ├── ner_finetuning.py            # NER model training
│   ├── layoutlm_finetuning.py       # LayoutLM training
│   ├── relation_extraction_component.py
│   └── evaluate_models.py           # Model evaluation
│
├── models/                           # Trained model storage
│   ├── ner_legal_bert_insurance/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── layoutlmv3_insurance_forms/
│   └── relation_extractor_spacy/
│
├── data/                            # Data directory
│   ├── raw/                         # Original documents
│   │   └── sample_policy.pdf
│   ├── annotated/                   # Training data
│   │   ├── ner_annotations.json
│   │   └── relation_annotations.json
│   └── processed/                   # Output files
│
├── tests/                           # Unit tests
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_ner.py
│
├── docs/                            # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── model_training_guide.md
│
├── logs/                            # Application logs
│   └── app.log
│
├── temp/                            # Temporary files
└── uploaded_files/                  # User uploads
```

---

## ⚙️ Configuration

The `config.yaml` file controls all system settings. Key configurations:

### Model Paths
```yaml
model_paths:
  ner: "./models/ner_legal_bert_insurance/"
  layout: "./models/layoutlmv3_insurance_forms/"
  relations: "./models/relation_extractor_spacy/"
```

### Entity Types to Extract
```yaml
ner_settings:
  entity_labels:
    - INSURED_PARTY
    - POLICY_NUMBER
    - EFFECTIVE_DATE
    - COVERAGE_LIMIT
    - DEDUCTIBLE
    - PREMIUM
    - INSURER
```

### OCR Settings
```yaml
ocr_settings:
  deskew: true              # Correct crooked scans
  force_ocr: true           # Force OCR even if text exists
  language: "eng"           # Tesseract language code
```

### Performance Settings
```yaml
performance:
  use_gpu: true             # Enable GPU acceleration
  batch_size: 8             # Processing batch size
  num_workers: 4            # Parallel processing threads
```

For complete configuration options, see [config.yaml](config.yaml).

---

## 💻 Usage

### Running the Web Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload Document**: Click "Choose an insurance policy PDF" and select your file
2. **Wait for Processing**: The system will automatically:
   - Extract text from the PDF
   - Apply OCR if needed
   - Run NER models
   - Extract relationships
3. **View Results**: Navigate through tabs:
   - **Key Information**: Visual entity highlighting
   - **Structured Data**: Complete JSON output
   - **Relationships**: Extracted connections
   - **Raw Text**: Full document text

### Command-Line Usage

```python
from pipeline import PolicyAnalyzer, load_config

# Load configuration
config = load_config("config.yaml")

# Initialize analyzer
analyzer = PolicyAnalyzer(config)

# Analyze a document
result = analyzer.analyze_document("path/to/policy.pdf")

# Access results
print(f"Entities found: {len(result['entities'])}")
print(f"Relations found: {len(result['relations'])}")

# Save results
import json
with open("analysis_result.json", "w") as f:
    json.dump(result, f, indent=2)
```

### Batch Processing

```python
import os
from pathlib import Path

# Process all PDFs in a directory
pdf_dir = Path("data/raw")
results = []

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    result = analyzer.analyze_document(str(pdf_file))
    results.append(result)

# Save batch results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 🎓 Model Training

### Prerequisites for Training
- Annotated training data in IOB format
- GPU recommended (training can take hours on CPU)
- Minimum 1000 annotated examples per entity type

### Training the NER Model

```bash
cd scripts
python ner_finetuning.py --data ../data/annotated/ner_training.json \
                         --output ../models/ner_legal_bert_insurance/ \
                         --epochs 5 \
                         --batch_size 8
```

### Training the LayoutLM Model

```bash
python layoutlm_finetuning.py --data ../data/annotated/form_annotations/ \
                              --output ../models/layoutlmv3_insurance_forms/ \
                              --epochs 10
```

### Training the Relation Extraction Component

```bash
python relation_extraction_component.py --data ../data/annotated/relations.json \
                                       --output ../models/relation_extractor_spacy/
```

### Data Annotation

Use one of these tools to create training data:
- **Doccano**: https://github.com/doccano/doccano
- **Label Studio**: https://labelstud.io/
- **Prodigy**: https://prodi.gy/

See [docs/model_training_guide.md](docs/model_training_guide.md) for detailed instructions.

---

## 📚 API Reference

### PolicyAnalyzer Class

```python
class PolicyAnalyzer:
    def __init__(self, config: Dict[str, Any])
    def analyze_document(self, file_path: str) -> Dict[str, Any]
```

#### analyze_document()
Performs end-to-end analysis of an insurance policy document.

**Parameters:**
- `file_path` (str): Path to the PDF document

**Returns:**
- `dict`: Analysis results containing:
  ```python
  {
      "file_path": str,
      "ocr_performed": bool,
      "content": dict,           # Page-by-page text
      "entities": list,          # Extracted entities
      "relations": list,         # Extracted relationships
      "metadata": dict           # Processing metadata
  }
  ```

### DocumentProcessor Class

```python
class DocumentProcessor:
    def __init__(self, min_text_length_threshold: int = 100)
    def process_document(self, file_path: str) -> Dict[str, Any]
```

#### process_document()
Ingests and processes PDF documents with intelligent OCR routing.

**Parameters:**
- `file_path` (str): Path to the PDF document

**Returns:**
- `dict`: Document content with text and layout information

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. ModuleNotFoundError: No module named 'ingestion'

**Solution:**
```bash
# Ensure you're in the correct directory
cd /path/to/insurance-policy-analyzer

# Verify file exists
ls -l ingestion.py

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Tesseract not found

**Solution:**
```bash
# Windows
setx /M PATH "%PATH%;C:\Program Files\Tesseract-OCR"

# Verify installation
tesseract --version
```

#### 3. CUDA out of memory error

**Solution:**
Update `config.yaml`:
```yaml
performance:
  use_gpu: false  # Disable GPU
  batch_size: 2   # Reduce batch size
```

#### 4. Model loading takes too long

**Solution:**
Models are loaded once at startup. Use caching:
```python
@st.cache_resource
def load_analyzer(config_path):
    return PolicyAnalyzer(load_config(config_path))
```

#### 5. Poor extraction quality

**Solutions:**
- Ensure documents are not encrypted
- Check image quality for scanned documents (minimum 300 DPI)
- Fine-tune models on your specific document types
- Adjust confidence thresholds in `config.yaml`

### Getting Help

- **GitHub Issues**: https://github.com/your-username/insurance-policy-analyzer/issues
- **Documentation**: See `docs/` folder
- **Community Forum**: [Link to forum]

---

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
docker build -t insurance-analyzer .
docker run -p 8501:8501 -v $(pwd)/models:/app/models insurance-analyzer
```

### AWS Deployment

```bash
# Deploy to AWS ECS
aws ecs create-cluster --cluster-name insurance-analyzer-cluster
aws ecr create-repository --repository-name insurance-analyzer

# Push Docker image
docker tag insurance-analyzer:latest {account-id}.dkr.ecr.{region}.amazonaws.com/insurance-analyzer
docker push {account-id}.dkr.ecr.{region}.amazonaws.com/insurance-analyzer
```

### API Server (FastAPI)

For production REST API, use FastAPI:
```bash
pip install fastapi uvicorn

# Run API server
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `pytest tests/`
5. **Commit**: `git commit -m 'Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Include type hints
- Write unit tests for new features

### Areas for Contribution

- 🐛 Bug fixes
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧪 Additional test coverage
- 🌐 Multi-language support
- 📊 New visualization features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** - Transformers library and pre-trained models
- **spaCy** - NLP framework and custom components
- **Streamlit** - Rapid web app development
- **PyMuPDF** - PDF processing library
- **OCRmyPDF** - OCR enhancement pipeline
- **Legal-BERT** - Domain-specific BERT model by AUEB NLP Group

---

## 📞 Contact

**Project Maintainer**: Your Name
- Email: your.email@example.com
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

**Organization**: Your Organization
- Website: https://your-website.com
- Twitter: [@your-org](https://twitter.com/your-org)

---

## 📊 Project Status

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-yellowgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()

**Current Version**: v1.0.0  
**Last Updated**: November 2025  
**Status**: Active Development

---

## 🗺️ Roadmap

### Version 1.1 (Q1 2026)
- [ ] Multi-language support (Spanish, French, German)
- [ ] Enhanced form extraction (ACORD 25, 126, 130)
- [ ] Integration with popular insurance platforms

### Version 1.2 (Q2 2026)
- [ ] Real-time collaboration features
- [ ] Advanced fraud detection models
- [ ] Automated policy comparison

### Version 2.0 (Q3 2026)
- [ ] Generative AI Q&A with policies
- [ ] Policy drafting assistance
- [ ] Regulatory compliance checking

---

**Built with ❤️ for the Insurance Industry**
