import traceback
from ...intitalizer import FUNC_NAME
import pandas as pd

class SkillExtractor:

    def __init__(self, skills_weighted_model=None):
        self._skills_weighted_model = skills_weighted_model

    @classmethod
    def clean_entities(cls, ner_entities):
        try:
            skills_ner_tuples = list(filter(lambda x: x[0] == "SKILLS", ner_entities))
            skills_ner_tuples = list(map(lambda x: x[1], skills_ner_tuples))
            skills_ner_tuples = list(map(lambda x: x.upper().replace(",", " ").strip(), skills_ner_tuples))
            skills_ner_tuples = list(set(skills_ner_tuples))
            return skills_ner_tuples

        except Exception as e:
            e = traceback.format_exc()
            return ner_entities

    def fn_get_skills_using_pretrained_weighted_vector(self, list_of_texts):
        found_skills_list = []
        try:
            if not any(list_of_texts):
                return found_skills_list
            else:
                total_text = len(list_of_texts)
                if isinstance(list_of_texts[0], str):
                    list_of_texts = [list_of_texts]

                text = " ".join(list_of_texts[0])
                df_skills = pd.DataFrame(self._skills_weighted_model.transform([text]).T.todense(),
                                         index=self._skills_weighted_model.get_feature_names_out(),
                                         columns=["tfidf"])
                df_skills = df_skills[df_skills.tfidf > 0]
                df_skills = df_skills.sort_values(["tfidf"], ascending=False).reset_index()
                found_skills_list = list(df_skills["index"].values)
        except Exception as e:
            e = traceback.format_exc()
        return found_skills_list

    def fn_find_all_skills_text(self, list_of_texts):
        all_skills_list = []
        try:
            if list_of_texts is not None:
                all_skills_list.extend(self.fn_get_skills_using_pretrained_weighted_vector(list_of_texts))
                list_of_unique_skills_found = list(set(all_skills_list))
                return list_of_unique_skills_found
            else:
                return []
        except Exception as e:
            e = traceback.format_exc()
            return all_skills_list
