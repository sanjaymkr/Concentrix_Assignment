import traceback
from .intitalizer import entity_link_obj 
import pandas as pd
import re
 
class CertificationExtractor:

    @classmethod
    def check_result(cls, val:dict) -> bool:
        return True if val['certificate_name'] or val['specialization'] or val['issuer'] or val['validity_date'] else False

    @classmethod
    def final_result_check(cls, result: list) -> list:
        if result:
            result = list(filter(CertificationExtractor.check_result, result))
            if result:
                return result
        return []

    @classmethod
    def fn_get_certification_data(cls,ents):
        clustered_entity = entity_link_obj.get_grouped_entities(ents)
        filtered_cert = list(clustered_entity.values())
        cert_dict_list=[]
        for cert in filtered_cert:
            cert_dict= {
                'certificate_name': '',
            'issuer' : '',
            'specialization' : '',
            'validity_date' : '',
                }
            for j in cert:
                if j[0]=='CERTIFICATION_NAME':
                    cert_dict['certificate_name']=j[1]
               
                elif j[0]=='FIELD':
                    cert_dict['specialization']=j[1]
                elif j[0]=="ISSUER":
                    cert_dict['issuer']=j[1]
                elif j[0]=="VALIDITY":
                    cert_dict['validity_date']=j[1]
                else:
                    continue
            cert_dict_list.append(cert_dict)
        cert_dict_list = CertificationExtractor.final_result_check(cert_dict_list)
        return cert_dict_list