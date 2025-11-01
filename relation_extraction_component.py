from transformers import NllbMoePreTrainedModel
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetDepthwiseConv2d
import spacy 
from spacy.tokens import Doc, Span
from spacy.training import Example
import random
from spacy.language import Language

# 1. Defining Custom Component
class RelationExtractor:
    def __init__(self, nlp: Language):
        """Initializes the relation extraction component"""
        if not Doc.has_extension('relations'):
            Doc.set_extension("relations", default=[])
        self.nlp = nlp
        self.model = self._load_dummy_model()
    
    def _load_dummy_model(self):
        # Placeholder for a real ML model. This dummy model looks for specific keywords.
        def dummy_model(doc):
            relations = []
            for ent1 in doc.ents:
                for ent2 in doc.ents:
                    if ent1.start == ent2.start:
                        continue
                    # Simple rule: if 'issued to' is between an INSURER and INSURED_PARTY
                    if ent1.label_ == "INSURER" and ent2.label_ == "INSURED_PARTY":
                        # Check text between entities
                        inter_text = doc.text[ent1.end_char:ent2.start_char].lower()
                        if "issued to" in inter_text:
                            relations.append({"head": ent1.text, "child": ent2.text, "label": "ISSUED_TO"})
            return relations
        return dummy_model
    
    def __call__(self, doc: Doc):
        """
        This method is called when the component is used in the pipeline.
        It finds entities and predicts relations between them.
        """
        relations = self.model(doc)
        doc._.relations = relations
        return doc
  # 2. Training Data preparation
TRAIN_DATA = [
    (
        "Acme Inc. issued a policy to John Doe.",
        {
            # First, define the entities
            "entities": [
                (0, 9, "INSURER"),         # "Acme Inc."
                (30, 38, "INSURED_PARTY")  # "John Doe"
            ],
            # Now, define the relationship between them
            "relations": [
                (0, 9, 30, 38, "ISSUED_TO") # (Acme Inc.) ---ISSUED_TO---> (John Doe)
            ],
        },
    ),
    (
        "The policy for Jane Smith was from Liberty Mutual.",
        {
            "entities": [
                (16, 26, "INSURED_PARTY"), # "Jane Smith"
                (36, 50, "INSURER")        # "Liberty Mutual"
            ],
            # This example is great for training because the entities are reversed in the text.
            "relations": [
                (36, 50, 16, 26, "ISSUED_TO") # (Liberty Mutual) ---ISSUED_TO---> (Jane Smith)
            ],
        },
    ),
    (
        "The policy limit is $500,000 for the property.",
        {
            "entities": [
                (20, 27, "COVERAGE_LIMIT"),  # "$500,000"
                (36, 44, "COVERED_PROPERTY") # "property" (example label)
            ],
            # This is a "negative example".
            "relations": [], # It teaches the model that not all entities are related.
        }
    )
]
# 3. training the component (Conceptual)
def train_relation_component():
    """
    This function outlines the training loop for the custom component.
    A full implementation requires a proper model architecture (e.g., using Thinc's chain).
    """
    nlp = spacy.blank('en')
    ner=nlp.add_pipe('ner')
    ner.add_label("INSURER")
    ner.add_label("INSURED_PARTY")
    ner.add_label("COVERAGE_LIMIT")
    rel_extractor= nlp.add_pipe("relation_extractor", after="ner")
     
    # Training would happen here. Because our model is a dummy, we skip the actual training loop.
    # A real training loop would look like this:
    # optimizer=nlp.begin_training()
    # for ith in range(n_iter):
        #random.shuffle(TRAIN_DATA)
        #losses={}
        #for text , annotations in TRAIN_DATA :
           # doc= nlp.make_doc(text)
           # example=Example.from_dict(doc , annotations)
           # nlp.update([example], drop = 0.5, losses=losses , sgd =optimizer)
        #  print(losses)
    print("Conceptual training complete. A dummy model is in place.")
    return nlp
    # 4. Example Usage
    if __name__ == '__main__' :
        #Train our model
        trained_nlp = train_relation_component()
         
        #Testing the pipeline
        test_text = "a policy was issued to jane Doe by the provider Acme insurance co."

        #Manually adding entities to the doc for the RE component to process,
        doc=trained_nlp.make_doc(test_text)
        ents = [
        # (start_char, end_char, "LABEL")
        doc.char_span(23, 31, label="INSURED_PARTY"), # "Jane Doe"
        doc.char_span(50, 68, label="INSURER")         # "Acme Insurance Co."
    ]
        doc.ents=ents
        rel_extractor= trained_nlp.get_pipe("relation_extractor")
        doc= rel_extractor(doc)
        print(f"\n---Relation Extraction results for : '{test_text}'---")
        print(f'Entites : {[(ent.text , ent.label_) for ent in doc.ents]}')
        print(f"Relations : {doc._.relations}")
        # A second example to show the dummy rule in action
        test_text_2 = "Acme Insurance co. issued to John Doe."
        doc2= trained_nlp.make_doc(test_text_2)
        ents2 = [
        doc2.char_span(0, 18, label="INSURER"),         # "Acme Insurance Co."
        doc2.char_span(30, 38, label="INSURED_PARTY")  # "John Doe"
    ]
    doc2.ents=ents2
    doc2= rel_extractor(doc2)
    print(f"\n--- Relation Extraction Results for: '{test_text_2}' ---")
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc2.ents]}")
    print(f"Relations: {doc2._.relations}")









        
