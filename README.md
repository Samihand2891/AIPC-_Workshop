# AIPC-_Workshop
Hackathon submission for iit bombay aipc hackathon 
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
