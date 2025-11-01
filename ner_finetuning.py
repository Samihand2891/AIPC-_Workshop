import os 
import torch
import json

from transformers.data import data_collator
from transformers.models.clvp import number_normalizer
from datasets import Dataset , Dataset , ClassLabel , Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification)


import numpy as np
from seqeval.metrics import classification_report

def create_dummy_dataset():
    data = [
        {"tokens": ["John", "Doe", "works", "at", "Microsoft", "."], "ner_tags": [1, 1, 0, 0, 2, 0]},
        {"tokens": ["The", "company", "is", "located", "in", "New", "York", "."], "ner_tags": [0, 0, 0, 0, 0, 3, 3, 0]},
        {"tokens": ["Coverage", "is", "provided", "by", "Acme", "Insurance", "Co", "."], "ner_tags": [0, 0, 0, 0, 2, 2, 2, 0]}
    ]
    # Converting To Hugging Face Dataset Format
    dataset = Dataset.from_list(data)
    return dataset

def main():
    # 1. Configuration (setting up core parameters for the object)
    MODEL_CHECKPOINT = "nlpaueb/legal-bert-base-uncased"
    MODEL_OUTPUT_DIR = "./models/ner_legal_bert_insurance"
    
    # Defining entity labels and there specified mapping
    label_list = [
        'O',                   # 0
        'B-INSURED_PARTY',     # 1
        'I-INSURED_PARTY',     # 2
        'B-POLICY_NUMBER',     # 3
        'B-EFFECTIVE_DATE',    # 4
        'I-EFFECTIVE_DATE',    # 5
        'B-DATE',              # 6
        'B-COVERAGE_LIMIT',    # 7
        'B-INSURER',           # 8
        'I-INSURER'            # 9
    ]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # Loading and preparing data for correct Hugging face dataset library format
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    raw_dataset = create_dummy_dataset()
    
    # defining features (defining schema of dataset), to align ner_tags with ClassLabel
    features = raw_dataset.features.copy()
    features["ner_tags"] = Sequence(feature=ClassLabel(names=label_list))
    raw_dataset = raw_dataset.cast(features)
    
    # Splitting into train and test data (70% train and 30% test)
    train_test_split = raw_dataset.train_test_split(test_size=0.3, seed=42)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    # Function to Solve mismatch of BERT tokenizer with labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # For subwords of same word, labeling them as first subword is possible
                    # or -100 can be used as it is a special value that the Hugging Face framework 
                    # automatically ignores when calculating the loss. This ensures these special tokens 
                    # don't confuse the model during training.
                    label_ids.append(label[word_idx] if True else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
        # This function is applied to every example using .map and using large no. of batches of example 
        tokenized_datasets= dataset.map(tokenize_and_align_labels, batched=True)


     # 3. Model and Trainer Setup
    model=AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
    )
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
    #Takes examples from tokenized datasets batch and intelligently combines them into a single batch to fed into a GPU

    def compute_metrics(p) :
        predictions , labels = p
        predictions = np.argmax(predictions , axis =2 )
        true_predictions = [
            label_list[p]
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        true_labels = [
            label_list[l]
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        report= classification_report(true_labels, true_predictions , output_dict=True)
        results = {
            "precision" : report['microavg']['precision'],
            "recall" : report["micro avg"]["recall"] ,
            "f1": report["micro avg"]["f1-score"],
            "accuracy": report["accuracy"]
        }
        return results
    
    training_args = TrainingArguments(
            output_dir = MODEL_OUTPUT_DIR,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5, #Can increase to 50 for major datasets
            weight_decay=0.01 ,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,

        )
    
    trainer=Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets['test'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    print("Starting NER model fine tuning")
    trainer.train()
    print("Fine tuning complete")
    trainer.save_model(MODEL_OUTPUT_DIR)
    trainer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to {MODEL_OUTPUT_DIR}")
    from transformers import pipeline
    ner_pipeline = pipeline(
        "ner",
        model=MODEL_OUTPUT_DIR,
        tokenizer=MODEL_OUTPUT_DIR,
        aggregation_strategy="simple"
    )

    test_text = "Coverage is provided by Acme Insurance Co. to John Doe."
    results = ner_pipeline(test_text)
    print("\n--INFERENCE RESULTS--")
    print(f"Text: {test_text}")
    for entity in results :
        print(f"Entity : {entity['word']}, Type : {entity['entity_group']} ,Score : {entity['score'] :.4f}" )


if __name__ == '__main__':
    main()