import concurrent.futures
import multiprocessing
import os
import re
import tempfile
import traceback
import unicodedata
from collections import ChainMap, defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Union

from thefuzz import fuzz

import numpy as np
import spacy
from cleantext import clean as clean_text
from PyPDF2 import PdfReader, PdfWriter
from ..intitalizer import ALL_NER_TAGS, FUNC_NAME, TEMP_REMOVE,CONTACT_NER
from ..utils.conversion import Conversion
from ..utils.convert2text import PDF2Text
from ..utils.education_table import DataDecoder
from ..utils.experience_table import Experience_table_extractor
from .entity_recognition.Contact.contact_extractor import ContactExtractor
from .entity_recognition.Contact.current_location import CurrentLocationExtractor
from .entity_recognition.model_reader import NER
from .entity_recognition.ner_main import NER_Main
from .prediction.block_classifier import BlockClassifier
from ..objectdetector import ObjectDetector
from .entity_recognition.experience_calc import total_experience_from_text
from .entity_recognition.experience_extractor import ExperienceExtractor

class PdfSegmenter:
    temp_dir = ""
    def __init__(self, tempfiles_dir: str, segment_model_path: list, segment_output_path: str, section_classifier_model_path : str, tokenizer_path : str, tf_bert_model_path : str) -> None:
        self.TEMPFILES_DIR = Path(tempfiles_dir)
        segment_model_path = [Path(x) for x in segment_model_path]
        segment_output_path = Path(segment_output_path)
        section_classifier_model_path = Path(section_classifier_model_path)
        tokenizer_path = Path(tokenizer_path)
        tf_bert_model_path = Path(tf_bert_model_path)
        self.current_loc_obj = CurrentLocationExtractor()
        if not segment_output_path.is_dir():
            raise ValueError("segment_output_path must be a valid directory path")


        if not tokenizer_path.is_dir():
            raise ValueError("tokenizer_path must be a valid directory path")

        if not tf_bert_model_path.is_dir():
            raise ValueError("tf_bert_model_path must be a valid directory path")

        if not self.check_segment_model(section_classifier_model_path, '.h5'):
            raise ValueError(f"Please checck {section_classifier_model_path}")

        self.check_tempfiles_folders()

        self.CONVERTED_PDF_PATH = os.path.join(self.TEMPFILES_DIR, "converted_pdf")
        self.TEMP_PDF_DIR = os.path.join(self.TEMPFILES_DIR, "pdf")
        self.TEMP_IMAGE_DIR = os.path.join(self.TEMPFILES_DIR, "images")


        self.segment_model_path = segment_model_path
        self.segment_output_path = segment_output_path


        self.detector_obj = ObjectDetector(model_path=self.segment_model_path, output_folder=self.segment_output_path)
        self.convert_file_obj = Conversion(self.CONVERTED_PDF_PATH)
        self.block_classifier_obj = BlockClassifier(section_classifier_model_path,tokenizer_path,tf_bert_model_path)
        self.pdf2text_obj = PDF2Text()
        self.ner = NER_Main(location_obj=self.current_loc_obj)
        self.ner_get_results = NER()
        self.decoder = DataDecoder()


    def check_tempfiles_folders(self):
        if not os.path.exists(self.TEMPFILES_DIR):
            os.makedirs(self.TEMPFILES_DIR, mode=0o777, exist_ok=True)

        sub_dirs = ["images", "pdf", "converted_pdf"]
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(self.TEMPFILES_DIR, sub_dir)
            if not os.path.exists(sub_dir_path):
                os.makedirs(sub_dir_path, mode=0o777, exist_ok=True)


    def check_segment_model(self, model_file, file_type):
        if not os.path.isfile(model_file):
            print('Invalid path:', model_file)
            return False
        elif os.path.splitext(model_file)[-1].lower() != file_type:
            print('Invalid file format:',
                  os.path.splitext(model_file)[-1].lower())
            return False
        else:
            print('Valid ONNX model file')
            return True


    def remove_files(self, paths: Union[str, List[str]]) -> None:
        try:
            if isinstance(paths, str):
                paths = [paths]

            for path in paths:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    try:
                        os.rmdir(path)
                    except OSError:
                        logger.error(f"[{__class__}] ({FUNC_NAME()}) Folder {path} is not empty, skipping deletion.")
                else:
                    print(f"Invalid path: {path}")
        except Exception as e:
            warning = ValueError("Invalid input. Please provide a single file path or a list of file paths.")
            logger.warn(f"[{__class__}] ({FUNC_NAME()}) warning : {warning}")
            logger.warn(f"[{__class__}] ({FUNC_NAME()}) error : {e}")


    def fn_read_clean_data(self,text_data):
        if text_data is not None and len(text_data) != 0:
            text_data = map(str.strip, text_data)
            text_data = map(
                lambda x: unicodedata.normalize("NFKD", x)
                .encode("ascii", "ignore")
                .decode("utf8")
                .strip(),
                text_data,
            )
            text_data = map(lambda x: clean_text(x, lower=False, no_emoji=True), text_data)
            text_data = list(filter(lambda x: x not in ["\n", ""], text_data))
            return text_data
        else:
            logger.info(f"[{__class__.__name__}] ({FUNC_NAME()}) text_data : {[]}")
            return []

    def process_raw_text(self,text:Union[List[str],str]):
        if isinstance(text,str):
            text = text.strip()
            clean_text = self.fn_read_clean_data([text])
        else:
            clean_text = self.fn_read_clean_data(text)
        return clean_text

    def get_table_edu(self,block_text):
        text=list(map(lambda record: {str(k).replace('\r', ''): str(v).replace('\r', '') for k, v in record.items()}, block_text))
        temp_result=self.decoder.decode_dict_data(text)
        return temp_result

    def get_entities(self,section,text):
        text = text.strip()
        clean_text = self.fn_read_clean_data([text])
        text = clean_text[0] if clean_text else text
        response = self.ner.recognize(section,text)
        return response

    def clean_file_name(self,path):
        saved_folder_name = os.path.splitext(os.path.basename(path))[0]
        saved_folder_name = os.path.splitext(os.path.basename(path))[0].lower()
        saved_folder_name = re.sub(r"\s+", r" ",saved_folder_name)
        saved_folder_name = re.sub("[^a-z\s]", repl="", string=saved_folder_name, flags=(re.I|re.M)).strip()
        saved_folder_name = re.sub(r"\s", r"_", saved_folder_name)
        logger.info(f"[{__class__.__name__}] ({FUNC_NAME()}) saved_folder_name : {saved_folder_name}")
        return saved_folder_name

    def save_response(self,response, file_path):
        self.clean_file_name(file_path)
        flatten_list = list(chain(*response))
        combined_dict = {}
        temp_exp_dict = {"total_experience": 0}
        for ele in flatten_list:
            for key, value in ele.items():
                if key == "total_experience":
                    temp_exp_dict[key] = value + temp_exp_dict[key] if value is not None else 0 + temp_exp_dict[key]
                else:
                    combined_dict.setdefault(key,[]).extend(value)
        # print(combined_dict)
        combined_dict = map(lambda x: (x[0],list(filter(lambda x: len(x)>0, x[1]))),combined_dict.items())
        combined_dict = dict(combined_dict)
        combined_dict['total_experience'] = temp_exp_dict["total_experience"]
        logger.info(f"[{__class__.__name__}] ({FUNC_NAME()}) combined_dict : {combined_dict}")
        return combined_dict

    def classify(self,text:str):
        classifed = self.block_classifier_obj.classify(text)
        classifed = {key:text for key in classifed.keys()}
        return classifed


    def list_string_block_classify(self, list_text:List[str]):
        list_text = np.array(list_text)
        vect = np.vectorize(self.classify, otypes=[np.ndarray])
        prediction = vect(list_text)
        return prediction
    def check_skills(self,RESULTS,ALL_TEXT):
        logger.info(f"[{__class__.__name__}] ({FUNC_NAME()}) Additional Skill Searching")
        try:
            skills = RESULTS.get('skills',[])
            ALL_TEXT_SKILLS = map(lambda x : self.ner.recognize("Skills",x),ALL_TEXT)
            ALL_TEXT_SKILLS = filter(
                lambda x : any(x.values()), ALL_TEXT_SKILLS
            )
            ALL_TEXT_SKILLS = list(set(chain.from_iterable(chain(*map(lambda x : (x.values()),ALL_TEXT_SKILLS)))))
            ALL_TEXT_SKILLS.extend(skills)
            ALL_TEXT_SKILLS = list(set(sorted([skill.casefold() for skill in ALL_TEXT_SKILLS],reverse=True)))
            RESULTS['skills'] = ALL_TEXT_SKILLS
        except:
            logger.warn(f"[{__class__.__name__}] ({FUNC_NAME()}) {traceback.format_exc()}")
        return RESULTS

    def map_tables_headings(self, edu_table_dict_list:list):

        try:
            if len(edu_table_dict_list) > 0 :
                dict_keys = list(edu_table_dict_list[0].keys())
                dict_keys =list(map(lambda d: re.sub("\s*", repl="", string=d), dict_keys))
                dict_keys =list(map(lambda d: re.sub(r"[\,\.?<>:';{}|()]", repl="", string=d), dict_keys))
            map_dict = {}
            for d in dict_keys:
                doc_ = self.edu_headings_model(d)
                ents = [ent.label_ for ent in doc_.ents]
                if len(ents) > 0 and set(doc_) <= set(ents):
                    map_dict[d] = ents[0]
                else:
                    continue

            logger.info(f"[{__name__}] ({FUNC_NAME()}) edu_heads map_dict: {map_dict}")
            new_dict_list = []
            if len(map_dict) != 0:
                for edu_dict in edu_table_dict_list:
                    # print(edu_dict)
                    temp_dict = {}
                    for i in range(0, len(dict_keys)):
                        if map_dict[dict_keys[i]] in temp_dict.keys():
                            temp_var1 = temp_dict[map_dict[dict_keys[i]]] # already stored result in key
                            temp_var2 = list(edu_dict.values())[i] #new coming value to be stored in same key
                            doc_1 = self.edu_model(text=temp_var1)
                            doc_2 = self.edu_model(text=temp_var2)
                            ents1 = [ent.label_ for ent in doc_1.ents]
                            ents2 = [ent.label_ for ent in doc_2.ents]
                            if map_dict[dict_keys[i]] in ents1 and map_dict[dict_keys[i]] not in ents2:
                                pass # keep ents1
                            elif map_dict[dict_keys[i]] in ents2 and map_dict[dict_keys[i]] not in ents1:
                                temp_dict[map_dict[dict_keys[i]]] = list(edu_dict.values())[i] # save ents2
                            elif map_dict[dict_keys[i]] in ents1 and map_dict[dict_keys[i]] in ents2:
                                temp_dict[map_dict[dict_keys[i]]] = list(edu_dict.values())[i]# append both and save
                            else:
                                pass
                        else:
                            temp_dict[map_dict[dict_keys[i]]] = list(edu_dict.values())[i]
                    new_dict_list.append(temp_dict)
                return new_dict_list
            else:

                return edu_table_dict_list
        except Exception as e:
            e = traceback.format_exc()
            return edu_table_dict_list


    def check_missing_text(self,cv_text,parsed):
        all_text = ""
        for t in cv_text:
            if isinstance(t,list):
                all_text += r"\n".join(t)
            else:
                all_text += r"\n"+t
        all_text = all_text.strip()
        try:
            for key,val in parsed.items():
                if isinstance(val,str):
                    # print(key)
                    sub_strings = val.split("\n")
                    sub_strings = list(map(str,sub_strings))
                    for sub_string in sub_strings:
                        pattern = fr"{re.escape(sub_string.strip())}"
                        regex = re.compile(pattern, re.I)
                        all_text = regex.sub('', all_text)
                        for idx, text in enumerate(cv_text):
                            if isinstance(text,str):
                                text = regex.sub('', text).strip()
                                cv_text[idx] = [text]
                            else:
                                text = regex.sub('', text[0]).strip()
                                cv_text[idx] = [text]

                if key in ['contact_no','email','skills','date_of_birth','current_location',
                        'candidate_name','designation',"address","summary"]:
                    if isinstance(val,str):
                        sub_strings = val.split("\n")
                        sub_strings = list(map(str,sub_strings))
                        for sub_string in sub_strings:
                            pattern = fr"{re.escape(sub_string.strip())}"
                            regex = re.compile(pattern, re.I)
                            all_text = regex.sub('', all_text)
                            for idx, text in enumerate(cv_text):
                                if isinstance(text,str):
                                    text = regex.sub('', text).strip()
                                    cv_text[idx] = [text]
                                else:
                                    text = regex.sub('', text[0]).strip()
                                    cv_text[idx] = [text]
                    elif isinstance(val,list):
                        val = filter(None,val)
                        val = list(map(str,val))
                        for v in val:
                            pattern = re.compile(re.escape(v),re.I)
                            all_text = pattern.sub('', all_text)
                            for idx, text in enumerate(cv_text):
                                if isinstance(text,str):
                                    text = pattern.sub('', text).strip()
                                    cv_text[idx] = [text]
                                else:
                                    text = pattern.sub('', text[0]).strip()
                                    cv_text[idx] = [text]
                if key in ["education_details","work_exp","certifications","projects"]:
                    try:
                        dict_val = chain.from_iterable(map(lambda x : x.values(), val))
                        dict_val = filter(None,dict_val)
                        dict_val = list(map(str,dict_val))
                        for entity in dict_val:
                            pattern = f"{entity.strip()}"
                            regex = re.compile(pattern,re.I)
                            all_text = regex.sub('', all_text)
                            for idx, text in enumerate(cv_text):
                                if isinstance(text,str):
                                    text = pattern.sub('', text).strip()
                                    cv_text[idx] = [text]
                                else:
                                    text = pattern.sub('', text[0]).strip()
                                    cv_text[idx] = [text]
                    except:
                        pass
                if key in ["social_links"]:
                    dict_val = list(val.values())
                    dict_val = filter(None,dict_val)
                    dict_val = list(map(str,dict_val))
                    for v in dict_val:
                        pattern = f"{re.escape(v.strip())}"
                        regex = re.compile(pattern,re.I)
                        all_text = regex.sub('', all_text)
                        for idx, text in enumerate(cv_text):
                            if isinstance(text,str):
                                text = regex.sub('', text).strip()
                                cv_text[idx] = [text]
                            else:
                                text = regex.sub('', text[0]).strip()
                                cv_text[idx] = [text]

        except:
            logger.error(f"[{__class__.__name__}] ({FUNC_NAME()}) {traceback.format_exc()}")
        parsed["missing"] = {
            "text" : all_text.strip(), #cv_text
            "tags" : ALL_NER_TAGS
        }

        return parsed

    def get_designation(self, text_list, model, to_remove):
        def removal(word):
            score = fuzz.token_set_ratio(word,to_remove)
            return (word,score)

        check_fuzz_match = lambda words : fuzz.token_set_ratio(words[0].lower(),words[1].lower())
        exp_ner = map(lambda text: self.ner_get_results.get_results(text,model),text_list)
        designation_tuples = list(map(lambda x: x, exp_ner))
        designation_tuples = filter(None,designation_tuples)
        designation_tuples = chain(*designation_tuples)
        designation_tuples = filter(lambda x : x[0] == 'DESIGNATION',designation_tuples)
        designation_tuples = map(lambda x : x[1],designation_tuples)
        designation_tuples = map(lambda x : x.strip().split("\n"),designation_tuples)
        designation_tuples = chain(*designation_tuples)
        designation_tuples = list(designation_tuples)

        removal_vect = np.vectorize(removal)
        if designation_tuples:
            designation_tuples = removal_vect(designation_tuples)
            designation_tuples = np.rec.fromarrays(designation_tuples)
            designation_tuples = map(lambda x : x[0], filter(lambda x : x[1]<80, designation_tuples))
            designation_tuples = list(designation_tuples)
        return designation_tuples

    def remove_non_numeric(self,value):
        return re.sub(r'[^0-9.]', '', value) if isinstance(value, str) else value


    def get_text(self, file_path):
        """
        Predicts the tables and other respective cell contents in the given PDF file.

        Args:
        - pdf_file_path (str): Path to the input PDF file.

        """
        candidate_photo_path=''
        ALL_TEXT = []
        BORDER_TEXT = []
        BORDER_TEXT_RAW_VALUE=[]
        ALL_PDF_TEXT = self.pdf2text_obj.read_entire_pdf(file_path)

        doc = PdfReader(file_path)
        # Convert to Image
        image_paths = self.convert_file_obj.convert_and_return_image_path(file_path, self.TEMP_IMAGE_DIR)
        image_paths.sort()
        for page_num, page in enumerate(doc.pages):
            pdfWriter = PdfWriter()
            pdfWriter.add_page(page)
            with tempfile.NamedTemporaryFile(prefix=f"{page_num}_page_", suffix='.pdf', delete=False, dir=self.TEMP_PDF_DIR) as tmp_file:
                temp_pdf_file_path = tmp_file.name
                # Save Page
                pdfWriter.write(tmp_file)
                pdfWriter.close()
                tmp_file.close()
                # Image Path
                image_path = image_paths[page_num]
                # Prediction
                crop_file_paths, bboxes, photo_path = self.detector_obj.predict(image_path, False)
                if len(photo_path)>len(candidate_photo_path):
                    candidate_photo_path=photo_path
                part_bbox = bboxes.get('part',[])
                table_border = (bboxes.get('table_borders',[]))
                table_no_borders = bboxes.get('table_no_borders',[])
                if part_bbox:

                    sorted_bboxes = sorted(part_bbox, key=lambda x: (x[1], x[0]))
                    # Read Text
                    text = self.pdf2text_obj.read_text(temp_pdf_file_path, sorted_bboxes)
                    if (text is None) or (len(text)==0):
                        raise TypeError("Non Searchable PDF")
                    processed_raw_text = [self.process_raw_text(t) for t in text]
                    ALL_TEXT.extend(processed_raw_text) 
                if table_border:
                    sorted_bboxes = sorted(table_border, key=lambda x: (x[0], x[1]))
                    ls_text = self.pdf2text_obj.read_table_text(temp_pdf_file_path, sorted_bboxes)
                    cls_text = None
                    for text in ls_text:
                        cls_text="\n".join([str(key) for key in text[0].keys()]) + " ".join([str(value) for d in text for value in d.values() if value])
                        processed_raw_text = [cls_text]

                    if cls_text is None:
                        processed_raw_text = ls_text
                    BORDER_TEXT.extend(processed_raw_text)
                    BORDER_TEXT_RAW_VALUE.extend(ls_text)

                if table_no_borders:
                    sorted_bboxes = sorted(table_no_borders, key=lambda x: (x[0], x[1]))
                    no_border_text = self.pdf2text_obj.read_no_border_text(temp_pdf_file_path, sorted_bboxes)
                    ALL_TEXT.extend(no_border_text)
        ALL_TEXT = [block for block in ALL_TEXT if len(block) > 0]
        BORDER_TEXT = [block for block in BORDER_TEXT if len(block) > 0]
        return ALL_TEXT, ALL_PDF_TEXT, BORDER_TEXT, BORDER_TEXT_RAW_VALUE, candidate_photo_path


    def predict_tabular(self, all_text, border_raw_text):
        try:
            table_edu=[]
            experience_table = []

            all_text_dict = defaultdict(str)

            for text in border_raw_text:
                cls_text=" ".join([str(key) for key in text[0].keys()])+" ".join([str(value) for d in text for value in d.values() if value])

                for key in self.classify(cls_text).keys():
                    if key=="Education":
                        temp_edu = self.map_tables_headings(text) # Mapping to respective heading using NER
                        edu_table_data= self.get_table_edu(temp_edu)
                        logger.info(f"[{__name__}] ({FUNC_NAME()}) edu_table_data : {edu_table_data}")
                        table_edu.extend(edu_table_data)
                        if not edu_table_data:
                            processed_raw_text = [cls_text]
                            all_text.extend(processed_raw_text)
                    elif key =='Experience':
                        temp_table = Experience_table_extractor.decode_dict_data(text)
                        exp_table_data = list(map(ExperienceExtractor.experience_dates_mapping,temp_table))
                        exp_table_data = self.output_dict_filter.filter_dicts(
                            important_keys=['DESIGNATION', 'DATE', 'ORG'],
                            dict_list=exp_table_data,
                            min_key_value_count=2
                        )
                        experience_table.extend(exp_table_data)
                        if not exp_table_data:
                            processed_raw_text = [cls_text]
                            all_text.extend(processed_raw_text)
                    else:
                        processed_raw_text = [cls_text]
                        all_text.extend(processed_raw_text)
        except:
            logger.info(f"[{__class__.__name__}] ({FUNC_NAME()}) {traceback.format_exc()}")
        return  all_text, table_edu, experience_table


    def predict(self, text, temp_text, all_pdf_text, border_raw_text, candidate_photo_path, result_queue):
        try:
            table_edu=[]
            experience_table = []
            RESULTS = []
            ALL_TEXT = all_pdf_text
            temp_candidate_name=[]
            all_text_dict = defaultdict(str)

            gender_ = re.findall(r'\b(male|female|Male|Female|MALE|FEMALE)\b',temp_text, re.IGNORECASE)
            date_of_birth = ContactExtractor.date_of_birth_from_all_text(temp_text)
            email = re.findall(r"[A-z][a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", temp_text, re.IGNORECASE)
            mobile_re = re.findall(r"(?:\+?\d{1,3}\s?)?(\d{10})|\d{10}|(?<!\d)(\+91)\s\d{3}\s\d{3}\s\d{4}(?!\d)|^\+91\s\d{5}\s\d{5}$", temp_text,re.IGNORECASE)
            linkedin_url = re.findall(r"linkedin\.com\/in\/[a-zA-Z0-9-]+",temp_text,re.IGNORECASE)

            text, table_edu, experience_table = self.predict_tabular(text,border_raw_text)
            # each_block_word_count = [block[0].split(" ") for block in text if len(block) > 0]
            each_block_word_count = [block[0].split(" ") for block in text if len(block) > 0]
            less_than_x_index = [i for i in range(0, len(each_block_word_count)) if len(each_block_word_count[i])<=4]



            ALL_TEXT_VECT = np.vectorize(self.list_string_block_classify,otypes=[np.ndarray])
            ALL_TEXT_PRED = ALL_TEXT_VECT(text)
            ALL_TEXT_PRED = np.hstack(ALL_TEXT_PRED)
            for arr in ALL_TEXT_PRED:
                if isinstance(arr,np.ndarray):
                    arr = arr.take(0)
                for key,val in arr.items():
                    all_text_dict[key] = all_text_dict[key]+" "+val
            for section,text in all_text_dict.items():
                text = text.strip()
                if section == "Extra":
                    continue
                # if section in ["Profile_Summary","Objective"]:
                if section == "Profile_Summary":
                    RESULTS.append({"professional_summary":text})
                elif section == "Objective":
                    RESULTS.append({"summary":text})
                elif section == "Certification":
                    RESULTS.append({"certifications":self.ner.recognize(section,text)})
                else:
                    entity_response = self.get_entities(section,text)
                    if entity_response is not None and len(entity_response) > 0:
                        is_empty = any(entity_response.values())
                        is_empty = not is_empty
                        RESULTS.append(entity_response)

            RESULTS = dict(ChainMap(*RESULTS))
            if 'gender' not in RESULTS.keys():
                if len(gender_) > 0:
                    RESULTS['gender'] = gender_[0]
                else:
                    RESULTS['gender'] = ''

            # Tabular Education Details
            if RESULTS.get('education_details'):
                RESULTS['education_details'].extend(table_edu)
            else :
                RESULTS.update({'education_details': table_edu})

            if 'email' not in RESULTS.keys():
                RESULTS['email'] = ContactExtractor.post_preocessing_on_email(email)

            if 'date_of_birth' in RESULTS.keys():
                RESULTS['date_of_birth'] = RESULTS.get('date_of_birth')
            elif date_of_birth and  'date_of_birth' not in RESULTS.keys() and ContactExtractor.check_dob(date_of_birth):
                RESULTS['date_of_birth']   = date_of_birth
            else:
                RESULTS['date_of_birth']  = []

            if 'candidate_name' not in RESULTS.keys():
                RESULTS['candidate_name'] = None
            if 'contact_no' not in RESULTS.keys():
                contact_no = ContactExtractor.extract_mobile_number1(mobile_re)
                RESULTS['contact_no'] = [] if contact_no is None else [contact_no]
            try:
                if not RESULTS.get('designation'):
                    ner_desig = []
                    for i in less_than_x_index:
                        text_temp = ALL_TEXT[i]
                        exp_ner = self.ner_get_results.get_results(text=text_temp, model=self.experience_model) # using only exp ner model for designation extraction
                        designation_tuples = filter(lambda x: x[0] == "DESIGNATION", exp_ner) # only tuples containing
                        designation_tuples = [i[1] for i in designation_tuples]
                        ner_desig.extend(designation_tuples)

                    if len(ner_desig)==0 and RESULTS.get('work_exp'):
                        all_temp_desig = [i["DESIGNATION"] for i in RESULTS.get('work_exp')]
                        if all_temp_desig:
                            all_temp_desig = list(filter(lambda x: x is not None and x!="", all_temp_desig))[0]
                        else:
                            all_temp_desig = ""
                        RESULTS['designation'] = all_temp_desig
                    else:
                        RESULTS['designation'] = ner_desig[0] if len(ner_desig)>0 else ""
            except Exception as e:
                e = traceback.format_exc()

            if RESULTS.get('work_exp'):
                RESULTS['work_exp'].extend(experience_table)
            else:
                RESULTS.update({'work_exp':experience_table})

            RESULTS['candidate_photo']=candidate_photo_path
            RESULTS['current_location'] = self.current_loc_obj.location_extr(results_dict=RESULTS)
            address = self.current_loc_obj.extract_city_substring(temp_text)
            if address and 'address' not in RESULTS.keys():
                RESULTS['address'] = address
            partial_address = self.current_loc_obj.address_extractor_from_all_text(temp_text,'any')
            if partial_address and 'address' not in RESULTS.keys():
                RESULTS['address'] = partial_address
            found_notice_period = self.notice_period_obj.extract_np(all_text=ALL_TEXT)
            if found_notice_period!=[]:
                RESULTS["notice_period"] = found_notice_period

            # Additional Skills for other sections
            RESULTS = self.check_skills(RESULTS,ALL_TEXT)
            candidate_name = RESULTS.get('candidate_name')
            if candidate_name is None or any(char.isdigit() for char in candidate_name or any(char.isascii() for char in candidate_name)):
                RESULTS = ContactExtractor.extract_candidate_name_block_len(ALL_TEXT, RESULTS,temp_candidate_name,
                                                                            self.spacy_en_core_web_sm_model,all_text_dict.get("Contact",None))
            candidate_name = RESULTS.get('candidate_name')
            if '\n' in candidate_name:
                candidate_name_parts = candidate_name.split('\n')
                if len(candidate_name_parts) >= 2:
                    candidate_name = candidate_name_parts[0]  # Update candidate_name

            total_exp = total_experience_from_text(temp_text)
            if total_exp:
                RESULTS['total_experience'] = total_exp
            else:
                RESULTS['total_experience'] = RESULTS.get('total_experience')
            if 'address' in RESULTS.keys() and RESULTS['address']:
                RESULTS['address'] = list(map(lambda x: ContactExtractor.postprocessing_on_address(RESULTS, x), RESULTS['address']))

            to_remove_from_designation = [candidate_name]
            designation = self.get_designation(ALL_TEXT,self.experience_model,to_remove_from_designation)

            RESULTS = self.check_missing_text(ALL_TEXT,RESULTS)
            result_queue.put({'inbuilt':RESULTS})

        except:
            return None