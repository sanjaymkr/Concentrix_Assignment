import gc
import asyncio
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import traceback
import threading
from typing import List
from .intitalizer import (BERT_MODEL_PATH, FUNC_NAME,
                           SEGMENTATION_MODEL_OUPUT_DIR,
                           SEGMENTATION_MODEL_PATH, TEMPFILES_DIR, TF_BERT_DIR,
                           TOKENIZER_DIR, logger)
from .entity_recognition.openai_parsing import Comparison_funcs
from .parser.mail_parallel import PdfSegmenter

compare_obj = Comparison_funcs()

segmenterObj = PdfSegmenter(tempfiles_dir=TEMPFILES_DIR,
                            segment_model_path=SEGMENTATION_MODEL_PATH,
                            segment_output_path=SEGMENTATION_MODEL_OUPUT_DIR,
                            section_classifier_model_path=BERT_MODEL_PATH,
                            tokenizer_path=TOKENIZER_DIR,
                            tf_bert_model_path=TF_BERT_DIR)


def fn_main(path):

    parser_result = {
        "summary":"",
        "professional_summary" : "",
        "email": [],
        "contact_no": [],
        "skills": [],
        "date_of_birth": [],
        "social_links": {
            "linkedin_url": "",
            "github": "",
            "medium": "",
            "hackerearth": "",
            "hackerrank": "",
            "other_blogs": "",
        },
        "address": [],
        "current_location": [],
        "education_details": [],
        "Achievement_details":[],
        "projects": [],
        "college": [],
        "certifications": [],
        "work_exp": [],
        "gender": "",
        "total_experience": None,
        "designation": "",
        "candidate_name":"",
        "notice_period":"",
        "extra" : [],
        "missing" : {}
        }


    try:
        op_parser_result = {}
        if not os.path.exists(path) and not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found")

        ACCEPTABLE_FILES = segmenterObj.convert_file_obj.get_file_type(path)
        if ACCEPTABLE_FILES in ["doc","docx"]:
            path = segmenterObj.convert_file_obj.convert_wordfile2pdf(path)
        ALL_TEXT, ALL_PDF_TEXT, BORDER_TEXT_RAW_VALUE, CANDIDATE_PHOTO_PATH = segmenterObj.get_text(path) # Parallel Process
        result_queue = Queue()
        temp_text = '\n'.join(ALL_PDF_TEXT)
  
        thread1 = threading.Thread(
            target=segmenterObj.predict,
            name="inbuilt_parser_thread",
            kwargs=({"text" : ALL_TEXT, 'temp_text': temp_text,
                    "all_pdf_text" : ALL_PDF_TEXT,
                    'border_raw_text': BORDER_TEXT_RAW_VALUE,
                    "candidate_photo_path" : CANDIDATE_PHOTO_PATH,
                    'result_queue': result_queue})
        )
        thread2 = threading.Thread(
            name="openai_parser_thread",
            kwargs=({"document_text" : temp_text,'result_queue':result_queue})
        )

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
    
        parsed = result_queue.get()
        parsed.update(result_queue.get())

        inbuilt_op = parsed.get("inbuilt",{})

        op_parser_result = compare_obj.combine_all_comparison_funcs(parser_dict=inbuilt_op, resume_text = ALL_PDF_TEXT)
        parser_result.update(op_parser_result)
    except Exception as e:
        e = traceback.format_exc()
    return parser_result
async def main_multiple(file_paths, loop):
 
    coros = [loop.run_in_executor(None, fn_main, path) for path in file_paths]
    results = await asyncio.gather(*coros)
    return results

def parse_multiple(file_paths:List[str]):
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    results = []
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main_multiple(file_paths, loop))
    loop.close()

    return results