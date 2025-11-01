#Code for ingesting pdf files and extracting there content using ..PyMupdf and ocrmypdf
import os , logging , tempfile

from sympy import true
from app import file_path, page_num
import fitz
import ocrmypdf
from typing import Dict , Any , Optional

#Configuuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

class DocumentProcessor :
    """ Creating a class for processing PDF documents , using PyMuPDF(fitz) we can get fast text extraction for digital native documents
    while using OCRmyPDF we can extract scanned images files performin robust OCR"""

    def __init__(self , min_text_length_threshold: int=100):
        self.min_text_length_threshold=min_text_length_threshold
        """ Arg(min_text_length_threshold) :
        intializing min text length required for it to be considered a digital native and not activate OCR"""

    def process_document(self, file_path :str) -> Dict[str , Any] :
        '''Args:
        file_path(str) : path to PDF file
        
        Returns: 
        Dict[str , Any] : A dicitionary that contains processing status , if OCR was performed or not and extracted text content page by page'''

        #For file not found
        if not os.path.exists(file_path) :
            logging.error(f'File not found : {file_path}')
            raise FileNotFoundError(f'file does not exist: {file_path}')

        logging.info(f'Statrting processing for the document:{file_path} ')

        #Attempting direct text extraction for digital natie files
        direct_text_content=self._extract_text_with_pymupdf(file_path)

        #Checking for extracted text .. is sufficient 
        total_text_length=sum(len(page_data['text'])
        for page_data in direct_text_content.values()) 

        if total_text_length > self.min_text_length_threshold* len(direct_text_content) :
            logging.info("Sufficient text is extracted , Skip OCR")
            return {
                "file_path" : file_path, 
                "ocr_performance" : False ,
                "content" : direct_text_content
            }
        else : 
            logging.warning("Insufficient text extracted . need to perform OCR")
            try : 
                ocr_text_content = self.perform_ocr(file_path)
                return {
                    "file_path" : file_path,
                    "ocr_performed" : True,
                    "content": ocr_text_content

                }
            except Exception as e :
                logging.error(f'OCR processing failed for {file_path}:{e}')
                return {
                    "file_path" : file_path ,
                    "ocr_performed" : False,
                    "error" : f"OCR failed : {e}",
                    "content": direct_text_content
                    
                }
                #If extraction fails in OCR used the direct extraction result 

    def _extract_text_with_pymupdf(self, file_path: str) -> Dict[int, Dict[str, Any]]:
        """Args:
          file_path (str): The path to the PDF file.

           Returns:
               [Dict]: A dictionary where keys are page numbers and values
            are dictionaries containing the page's text and a list of
            text blocks with their bounding boxes."""
        document_content = {}
        try : 
            doc = fitz.open(file_path)
            for page_num in range(len(doc)) :
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")
                page_text = page.get_text()
                #Structuring the extracted data
                structured_blocks = []
                for b in blocks :
                    x0 , y0 ,x1 , y1 , text , block_no , block_type = b
                    structured_blocks.append({
                        "bbox" : [x0 , y0 , x1 , y1],
                        "text" : text.strip(),
                        "block_type" : "text" if block_type == 0 else "image"
                    })
                document_content[page_num +1] = {
                    "text": page_text,
                    "blocks" : structured_blocks
                }
            doc.close()
        except Exception as e :
            logging.error(f"Error extracting text with PyMuPDF from {file_path} : {e}")
        return document_content

    def perform_ocr(self, input_file: str) -> Optional[Dict[int, Dict[str, Any]]]:
        """ Performs OCR on a PDF file using the OCRmyPDF library.
        It creates a temporary file for the output and then extracts text from it.

        Args:
            input_file (str): The path to the input PDF file.

        Returns:
            Optional[Dict]: The extracted text content after OCR,
            or None if the process fails."""
        
        # Creating a temporary file for OCR output
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_output_file:
            output_file_path =tmp_output_file.name
            try:
                logging.info(f"Running OCRmyPDF on {input_file}. Output will be saved to {output_file_path}")
                # Here, we create a new PDF and then use PyMuPDF to maintain a consistent data structure.
                ocrmypdf.ocr(
                    input_file,
                    output_file_path,
                    deskew=True,
                    force_ocr=True,  # Forcing OCR even if text is present
                    skip_text=True
                )
                logging.info("OCRmyPDF completed successfully")
                # Extracting text from newly created searchable PDF
                return self._extract_text_with_pymupdf(output_file_path)
            except Exception as e:
                logging.error(f'Cannot process file with OCR: {input_file}: {e}')
                raise
            finally:  # Cleaning up temporary files
                if os.path.exists(output_file_path):
                    os.remove(output_file_path)
