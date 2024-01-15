import inspect
import os

from dotenv import load_dotenv

from .utils.concentrix_config import Config
from .utils.entity_linking import EntityLinking

load_dotenv()


def FUNC_NAME():
    return inspect.stack()[1][3]


# CONFIG
PATH = os.getenv("ML_CONFIG_PATH")
config_obj = Config(path=PATH)


SEGMENTATION_MODEL_PATH: list = config_obj.get_segmentation_model_path()
SKILLS_WEIGHATGE_MODEL_PATH = config_obj.get_skills_weightage_model_path()
TF_BERT_DIR = config_obj.get_tfbert_model_path()
TOKENIZER_DIR = config_obj.get_tokenizer_model_path()
BERT_MODEL_PATH = config_obj.get_bert_model_path()
SKILLS_NER = config_obj.get_skill_ner_model()
EDUCATION_NER = config_obj.get_education_ner_model()
CONTACT_NER = config_obj.get_contact_ner_model()
EXPERIENCE_NER = config_obj.get_experience_ner_model()
EDUCATION_HEADINGS_NER = config_obj.get_education_table_headings_ner_model()
CERTIFICATION_NER = config_obj.get_certification_ner_model()
ACHIEVEMENT_NER = config_obj.get_achievement_ner_model()
PROJECT_NER = config_obj.get_project_ner_model()
SPACY_EN_CORE_WEB_SM_MODEL = config_obj.get_spacy_en_core_web_sm_model_path()
# ENTITY LINKING
entity_link_obj = EntityLinking(FUNC_NAME)
# STATIC FILES
CITIES_JSON = config_obj.get_cities_json_path()
STATES_JSON = config_obj.get_states_json_path()
OPENAI_EXCEL = config_obj.read_openai_excel()

# DIRECTORIES
SEGMENTATION_MODEL_OUPUT_DIR = config_obj.get_segmentation_model_output_path()
TEMPFILES_DIR = config_obj.get_tempdir_path()
# TEMP_IMAGE_DIR = config_obj.get_tempfiles_image_path()
# TEMP_PDF_DIR = config_obj.get_tempfiles_pdf_path()
# CONVERTED_PDF_DIR = config_obj.get_tempfiles_converted_pdf_path()
# STATIC_VALUES
TEMP_REMOVE = config_obj.remove_temp_files()

# NER TAGS
EDU_TAG = config_obj.get_education_tag()
CONTACT_TAG = config_obj.get_contact_tag()
EXPERIENCE_TAG = config_obj.get_experience_tag()
PROJECTS_TAG = config_obj.get_projects_tag()
CERTIFICATION_TAG = config_obj.get_certification_tag()

ALL_NER_TAGS = config_obj.get_all_ner_tags()

TAG_MAPPING = {
    'Achievement' : ['Achievement'.upper()],
    'Certification' : CERTIFICATION_TAG,
    'Contact' : CONTACT_TAG,
    'Education' : EDU_TAG,
    'Experience' : EXPERIENCE_TAG,
    'Extra' : ['Extra'.upper()],
    'Objective' : ['Objective'.upper()],
    'Profile_Summary' : ['Profile_Summary'.upper()],
    'Project' : PROJECTS_TAG,
    'Skills' : ['Skills'.upper()],
}
# MODEL TRAINING
## Train Logger
TRAIN_LOG_DIR = config_obj.get_train_log_path()
LOGFILE_NAME = config_obj.get_train_logfile()
LOGGER_NAME = config_obj.get_train_logger_name()
## Model Data
BLOCK_MODEL_DATA = config_obj.get_block_data_dir_path()
NER_MODEL_DATA = config_obj.get_ner_data_dir_path()
## Trained Model
BLOCK_TRAINED_MODEL = config_obj.get_block_model_dir_path()
NER_TRAINED_MODEL = config_obj.get_ner_model_dir_path()
# NER Config Files
NER_BASE_CONFIGS = config_obj.get_ner_config_files_path()
