#A LAyoutLMV3 model is used  for information extraction from structured documents like forms.
#This part of code provides conceptual framework and code structure for fine-tuning

from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    training_args

)
from transformers.pipelines.document_question_answering import apply_tesseract
from datasets import load_dataset #Dataset in HuggingFace format
import torch
from PIL import Image , ImageDraw


#1. Configuration 
DATASET_ID = "nielsr/funsd-layoutlmv3" 
MODEL_CHECKPOINT = "microsoft/layoutlmv3-base"
MODEL_OUTPUT_DIR = "./models/layoutlmv3_insurance_forms"

def main() :
    # --- 2. Load Data and Processor ---
    # The dataset should have columns: 'id', 'words', 'bboxes', 'ner_tags', 'image_path'
    dataset=load_dataset(DATASET_ID)

    labels=dataset['train'].features['ner_tags'].feature.names
    id2label={k : v for k , v in enumerate(labels)}
    label2id = {v : k for k , v in enumerate(labels)}
     
    processor = LayoutLMv3Processor.from_pretrained(MODEL_CHECKPOINT , apply_ocr = False)
    def preprocess_data(examples) :
        images=[Image.open(path).convert('RGB') for path in examples['image_path']]
        # Converts images into RGB format
        words=examples['words']
        boxes=examples['bboxes']
        word_labels= examples['ner_tags']
        encoded_inputs = processor(images , words , boxes = boxes , word_labels= word_labels,
        padding="max_length", truncation=True)
        return encoded_inputs
         
        processed_dataset = dataset.map(preprocess_data , batched =True , remove_columns=datasets['train'].column_names)

        #Setting format for Pytorch
        processed_dataset.set_format(type='torch')

        #3. Model and trainer setup
        model=LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT,
            id2label=id2label,
            label2id=label2id
        )
        training_args= TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            max_steps=1000,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
             learning_rate=1e-5,
             evaluation_strategy='steps',
             eval_steps=200,
             load_best_model_at_end=True,
             metric_for_best_model='f1'

        )
        import numpy as np
from seqeval.metrics import classification_report, accuracy_score

# NOTE: This function assumes 'id2label' is defined in the
# scope where this function is called (it was in your main() function).

def compute_metrics_layoutlm(p):
    predictions, labels = p
    # Get the most likely prediction ID by finding the index with the highest logit score
    predictions = np.argmax(predictions, axis=2)

    # 'seqeval' expects a list of lists of strings (one list for each example)
    # We need to convert IDs to labels and filter out -100 tokens
    true_predictions = []
    true_labels = []

    # Loop over each example in the batch
    for prediction, label in zip(predictions, labels):
        batch_predictions = []
        batch_labels = []
        
        # Loop over each token in the example
        for (p_token, l_token) in zip(prediction, label):
            # Only evaluate tokens that are NOT -100
            if l_token != -100:
                batch_predictions.append(id2label[p_token])
                batch_labels.append(id2label[l_token])
        
        true_predictions.append(batch_predictions)
        true_labels.append(batch_labels)

    # Get the detailed classification report
    # output_dict=True returns a dictionary of metrics
    report = classification_report(true_labels, true_predictions, output_dict=True)
    
    # Extract the key "micro" average scores.
    results = {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
        "accuracy": accuracy_score(true_labels, true_predictions)
    }
    
    return results
    trainer=Trainer(
        model=model,
        arg=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
        compute_metrics=compute_metrics_layoutlm,
    )

    # 4. Train the Model
    print('starting LayoutLMv3 model fine-tuning..')
    trainer.train()
    print('Fine-tuning complete')

    # Saving the model
    trainer.save_model(MODEL_OUTPUT_DIR)
    processor.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model and processor saved to {MODEL_OUTPUT_DIR}")

    # 6. Inference Example
    model_fine_tuned= LayoutLMv3ForTokenClassification.from_pretrained(MODEL_OUTPUT_DIR)
    processor_fine_tuned = LayoutLMv3Processor.from_pretrained(MODEL_OUTPUT_DIR, apply_ocr=False)
    image_path=dataset['test']['image_path'][0]
    image=Image.open(image_path).convert('RGB')


    # Use the processor to get words and boxes (as if from an external OCR)
    from transformers.models.layoutlmv3.processing_layoutlmv3 import apply_tesseract
    words , boxes = apply_tesseract(image , lang='eng')
    encoding = processor_fine_tuned(image , words , boxes=boxes , return_tensors='pt')

    #Ineference
    with torch.no_grad() :
        outputs= model_fine_tuned(**encoding)
        predicitions= output.logits.argmax(-1).squeeze().tolist()

        #Map predictions to labels
        token_boxes = encoding.bbox.squeeze().tolist()
        print("\n--- LayoutLMv3 Inference Results ---")
    for token_id, box, pred_id in zip(encoding.input_ids.squeeze().tolist(), token_boxes, predictions):
        token = processor_fine_tuned.tokenizer.decode([token_id])
        label=id2label[pred_id]
        if label != 'O' :
             print(f"Token: {token}, Label: {label}, BBox: {box}")

             
if __name__ == "__main__":
    print("This script is for demonstration purposes.")
     # main() # Uncomment to run with proper setup
 # This is a conceptual script. Running it requires significant setup
    # and a properly formatted dataset.


    
       
