# pipeline.py
# This module contains the PolicyAnalyzer class, which orchestrates the entire
# document analysis pipeline from ingestion to structured output.

import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline as hf_pipeline
)
from ingestion import DocumentProcessor
import spacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PolicyAnalyzer:
    """
    The main orchestrator class for the insurance policy analysis pipeline.
    It loads all necessary models and coordinates the analysis workflow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PolicyAnalyzer by loading all required models.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model paths
                                    and other settings.
        """
        self.config = config
        logging.info("Initializing PolicyAnalyzer...")
        
        # Initialize the document processor for ingestion
        self.document_processor = DocumentProcessor()
        
        # Load NER model
        self.ner_pipeline = self._load_ner_model()
        
        # Load relation extraction component (if available)
        self.relation_extractor = self._load_relation_extractor()
        
        logging.info("PolicyAnalyzer initialization complete.")
    
    def _load_ner_model(self) -> Optional[Any]:
        """
        Loads the fine-tuned NER model from the specified path.
        
        Returns:
            A Hugging Face pipeline for token classification (NER).
        """
        try:
            ner_model_path = self.config.get('model_paths', {}).get('ner')
            if not ner_model_path or not Path(ner_model_path).exists():
                logging.warning(f"NER model not found at {ner_model_path}. NER will be disabled.")
                return None
            
            logging.info(f"Loading NER model from {ner_model_path}")
            
            # Loading the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
            model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
            
            # Create a pipeline for easy inference
            ner_pipeline = hf_pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logging.info("NER model loaded successfully.")
            return ner_pipeline
            
        except Exception as e:
            logging.error(f"Error loading NER model: {e}")
            return None
    
    def _load_relation_extractor(self) -> Optional[Any]:
        """
        Loads the relation extraction component (spaCy model with custom component).
        
        Returns:
            A spaCy nlp object with the relation extraction component, or None.
        """
        try:
            re_model_path = self.config.get('model_paths', {}).get('relations')
            if not re_model_path or not Path(re_model_path).exists():
                logging.warning(f"Relation extraction model not found at {re_model_path}. RE will be disabled.")
                return None
            
            logging.info(f"Loading relation extraction model from {re_model_path}")
            
            # Load the spaCy model with the custom relation extraction component
            nlp = spacy.load(re_model_path)
            
            logging.info("Relation extraction model loaded successfully.")
            return nlp
            
        except Exception as e:
            logging.error(f"Error loading relation extraction model: {e}")
            return None
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        
        logging.info(f"Starting analysis of document: {file_path}")
        
        # Step 1: Document Ingestion
        logging.info("Step 1: Document Ingestion")
        ingestion_result = self.document_processor.process_document(file_path)
        
        if not ingestion_result or 'content' not in ingestion_result:
            logging.error("Document ingestion failed.")
            return {
                "error": "Document ingestion failed",
                "file_path": file_path
            }
        
        # Step 2: Named Entity Recognition
        logging.info("Step 2: Named Entity Recognition")
        entities = self._extract_entities(ingestion_result['content'])
        
        # Step 3: Relation Extraction
        logging.info("Step 3: Relation Extraction")
        relations = self._extract_relations(ingestion_result['content'], entities)
        
        # Step 4: Compile structured output
        logging.info("Step 4: Compiling structured output")
        analysis_result = {
            "file_path": file_path,
            "ocr_performed": ingestion_result.get('ocr_performed', False),
            "content": ingestion_result['content'],
            "entities": entities,
            "relations": relations,
            "metadata": {
                "total_pages": len(ingestion_result['content']),
                "total_entities": len(entities),
                "total_relations": len(relations),
                "ner_model_used": self.config.get('model_paths', {}).get('ner', 'None'),
                "re_model_used": self.config.get('model_paths', {}).get('relations', 'None')
            }
        }
        
        logging.info(f"Analysis complete. Extracted {len(entities)} entities and {len(relations)} relations.")
        return analysis_result
    
    def _extract_entities(self, content: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts named entities from the document content using the NER model.
        
        Args:
            content (Dict[int, Dict[str, Any]]): Page-by-page document content.
        
        Returns:
            List[Dict[str, Any]]: List of extracted entities with metadata.
        """
        if not self.ner_pipeline:
            logging.warning("NER model not loaded. Skipping entity extraction.")
            return []
        
        all_entities = []
        
        for page_num, page_data in content.items():
            page_text = page_data.get('text', '')
            
            if not page_text.strip():
                continue
            
            try:
                # Run NER on the page text
                ner_results = self.ner_pipeline(page_text)
                
                # Add page number to each entity
                for entity in ner_results:
                    entity['page'] = page_num
                    all_entities.append(entity)
                    
            except Exception as e:
                logging.error(f"Error extracting entities from page {page_num}: {e}")
        
        return all_entities
    
    def _extract_relations(
        self, 
        content: Dict[int, Dict[str, Any]], 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts relationships between entities using the relation extraction model."""
        
        
        if not self.relation_extractor:
            logging.warning("Relation extraction model not loaded. Skipping relation extraction.")
            return []
        
        all_relations = []
        
        for page_num, page_data in content.items():
            page_text = page_data.get('text', '')
            
            if not page_text.strip():
                continue
            
            # Get entities for this page
            page_entities = [e for e in entities if e.get('page') == page_num]
            
            if len(page_entities) < 2:
                continue
            
            try:
                # Process the text with spaCy
                doc = self.relation_extractor(page_text)
                
                # The custom component adds relations to doc._.relations
                if hasattr(doc._, 'relations'):
                    page_relations = doc._.relations
                    
                    # Add page number to each relation
                    for relation in page_relations:
                        relation['page'] = page_num
                        all_relations.append(relation)
                        
            except Exception as e:
                logging.error(f"Error extracting relations from page {page_num}: {e}")
        
        return all_relations


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Initialize the analyzer
    analyzer = PolicyAnalyzer(config)
    
    # Analyze a sample document
    sample_doc_path = "data/sample_policy.pdf"
    
    if Path(sample_doc_path).exists():
        result = analyzer.analyze_document(sample_doc_path)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Document: {result['file_path']}")
        print(f"OCR Performed: {result['ocr_performed']}")
        print(f"Total Pages: {result['metadata']['total_pages']}")
        print(f"Total Entities: {result['metadata']['total_entities']}")
        print(f"Total Relations: {result['metadata']['total_relations']}")
        print("\nFirst 5 Entities:")
        for entity in result['entities'][:5]:
            print(f"  - {entity['word']} ({entity['entity_group']}) - Page {entity['page']}")
        print("\nFirst 5 Relations:")
        for relation in result['relations'][:5]:
            print(f"  - {relation}")
    else:
        print(f"Sample document not found at {sample_doc_path}")
        print("Please place a sample PDF in the data/ directory to test the pipeline.")