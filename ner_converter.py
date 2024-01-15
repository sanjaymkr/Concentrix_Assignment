from ...intitalizer import FUNC_NAME, trainlogger

from itertools import chain

class NerConvertors:
    def __init__(self) -> None:
        pass
    
    def convert_to_spacy_format(self,postions_dict):
        postions_dict = list(map(lambda x : (x[0],{'entities':x[1]}), postions_dict.items()))
        return postions_dict
    
    def get_position(self,text,ents):
        positions = []
        tag_positions = {}       
        text = text.lower()
        for tag, val in ents:
            val = val.strip()
            pos = text.find(val.lower())
            if pos > -1:
                positions.append((pos,pos+len(val),tag))
        tag_positions[text] = positions
        return tag_positions
    
    def ner_convertor_main(self,data):
        all_data = []
        for sample in data:
            text = sample.get('text')
            tags_list = sample.get('tags_list')
            position_data = self.get_position(text,tags_list)
            spacy_format = self.convert_to_spacy_format(position_data)
            all_data.append(spacy_format)
        all_data = list(chain(*all_data))
        return all_data