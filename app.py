from pandas import options
import streamlit as st
import os
import yaml
from pipeline import PolicyAnalyzer
from spacy import displacy

# Page Configuration
st.set_page_config(page_title="AI Health Insurance policy analyzer"
, page_icon="📃" , layout="wide")

# Constants
CONFIG_PATH = "config.yaml"
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok= True)


@st.cache_resource
def load_analyzer(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            analyzer=PolicyAnalyzer(config)
            return analyzer
    except FileNotFoundError:
        st.error(f"Configuration file not found at {config_path}. Please create it.")
        return None
    except Exception as e:
        st.error(f"Error loading the analysis pipeline: {e}")
        return None


def visualize_entities(doc_text, entities):
    #Format entities for displacy
    displacy_ents = []
    for entity in entities:
        start = doc_text.find(entity['word'])
        if start != -1:
            end = start + len(entity['word'])
            displacy_ents.append({
                'start': start,
                'end': end,
                'label': entity['entity_group']
            })
   
    #Dummy doc structure for displacy
    doc = {"text": doc_text, 
           'ents': sorted(displacy_ents, key=lambda k: k['start']),
           'title': None}
    
    #Colors for entity types
    colors =  {
        "INSURED_PARTY": "#85C1E9", "POLICY_NUMBER": "#FAD7A0", 
        "EFFECTIVE_DATE": "#A9DFBF", "COVERAGE_LIMIT": "#F5B7B1",
        "INSURER": "#D7BDE2"
    }
    options ={"ents": list(colors.keys()), "colors": colors}
    html= displacy.render(doc , style="ent" , manual=True, options=options , page= False)
    st.write(html , unsafe_allow_html= True)


#Main UI of the Application
st.title("AI Health Insurance 🧑‍⚕️ policy analyzer 🧐")
st.markdown("""
This is a AI powered insurance policy prototype. Using NLP concepts we
can extract key insurance policy documents . You can upload your choice of policy document in PDF format to get started.""" )

#Loading analysis pipeline
analyzer =load_analyzer(CONFIG_PATH)
if analyzer : 
    uploaded_file= st.file_uploader("Choose an insurance policy PDF", type="pdf")

    if uploaded_file is not None:
        file_path=os.path.join(UPLOAD_DIR , uploaded_file.name) # Saves uploaded file to a tempory location
        with open(file_path , "wb") as f :
            f.write(uploaded_file.getbuffer())
            st.success(f"SUccessfully uploaded '{uploaded_file.name}'")
 
        #Analyzing the document..
        with st.spinner("Analyzing document .. This may take a moment . OCR maybe required") :
            try :
                analysis_result=analyzer.analyze_document(file_path)
            except Exception as e:
                st.error(f"An error occured during analysis : {e}")
                analysis_result=None

analysis_result = locals().get('analysis_result', None)
if analysis_result:
    st.header("Analysis Results")
    #Different tabs for different results
    tab1 , tab2 , tab3 , tab4 = st.tabs(["Key Information Highlighted", "Extracted Structured Data", "Extracted Relationships", "Full Extracted Text"])
    with tab1: 
        st.subheader("Key Information Highlighted")
        page_1 = analysis_result.get('content', {}).get(1)
        if page_1:
            page_1_text = page_1.get('text', "")
            page_1_entities = [e for e in analysis_result.get('entities', []) if e.get('page') == 1]
            visualize_entities(page_1_text, page_1_entities)
        else:
            st.warning("No text or entities found on the first page to visualize.") 
    with tab2:
        st.subheader("Extracted Structured Data")
        st.json(analysis_result)

    with tab3:
        st.subheader("Extracted Relationships")
        if analysis_result.get('relations'):
            st.table(analysis_result['relations'])
        else:
            st.info("No specific relationships were extracted by the model.")

    with tab4:
        st.subheader("Full Extracted Text")
        full_text = ""
        for page_num, page_data in sorted(analysis_result.get('content', {}).items()):
            full_text += f"--- PAGE {page_num} ---\n{page_data.get('text','')}\n\n"
        st.text_area("Raw Text", full_text, height=400)

    try:
        os.remove(file_path)
    except Exception:
        pass
else:
    if analyzer is None:
        st.warning("The analysis pipeline could not be loaded . Please check the configs and logs")
