import traceback
from ...intitalizer import logger, FUNC_NAME, entity_link_obj

from .experience_calc import fn_calculate_exp, fn_find_dates
import re
from datetime import datetime 

class ExperienceExtractor:

    @classmethod
    def filter_exp_entities(cls,ents):
        response = []
        DEFAULT_VALUE = ""
        all_fields = ["DURATION", "COMPANY_NAME", "DESIGNATION", "LOCATION"]
        mapped = {"DURATION":"DATE",
                    "LOCATION":"GPE",
                    "COMPANY_NAME":"ORG",
                    "DESIGNATION" : "DESIGNATION",
                    False : "is_currently_working"}
        new_ents = []
        for item in ents:
            key, value = item
            if '\n' in value:
                value = value.split('\n')[0]  # Get the portion before \n
            new_ents.append((key, value)) 
        clustered_entity = entity_link_obj.get_grouped_entities(new_ents)
        clustered_entity = list(clustered_entity.values())
        for group in clustered_entity:
            keys = [val[0] for val in group]
            for field in all_fields:
                if field not in keys:
                    group.append((field,DEFAULT_VALUE))
            group.append(("is_currently_working",False))

        clustered_entity = map(dict,clustered_entity)

        for exp_dict in clustered_entity:
            temp = {mapped.get(key,key) : val for key,val in exp_dict.items()}
            response.append(temp)
        return response


    @classmethod
    def get_exp_dates(cls, ents):
        calculated_exp = None
        try:
            dates_ner_tuples = list(filter(lambda x: x[0] == "DURATION", ents)) # only tuples containing
            dates_ner_tuples = list(set(map(lambda x: x[1], dates_ner_tuples)))
            dates_ner_tuples = list(map(lambda x: re.sub('(\d+(\.\d+)?)', r' \1 ', x).strip(), dates_ner_tuples)) #Add space between digit and alpha
            found_dates = list(map(fn_find_dates, dates_ner_tuples))
            calculated_exp = fn_calculate_exp(found_dates)
            return calculated_exp

        except Exception as e:
            e = traceback.format_exc()
            return calculated_exp

    @classmethod
    def experience_dates_mapping(cls,result):
        dates = fn_find_dates(result['DATE'])
        if len(dates) >=2:
            from_date,to_date = dates[0],dates[1]
            result['FROM_DATE'],result['TO_DATE'] = from_date,to_date
            result['DATE'] = [f"{from_date}" , f"{to_date}"]
        elif len(dates)==1:
            from_date  = dates[0]
            result['FROM_DATE'] = dates[0]
            result['TO_DATE'] = ''
            result['DATE'] = [f"{from_date}"]
        else:
            result['DATE'] = [result['DATE']]
            result['FROM_DATE'] = ''
            result['TO_DATE'] = ''
        if result['TO_DATE'] == datetime.now().strftime("%d-%m-%Y"):
            result['is_currently_working'] = True           
        return result