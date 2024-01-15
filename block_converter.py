import pandas as pd

class BlockDataConvertor:
    def __init__(self) -> None:
        pass

    def convert_to_block_format(self,text_data):
        all_text_data = []
        for key,val in text_data.items():
            all_text_data.extend([(t,key) for t in val])
        return pd.DataFrame(all_text_data)        