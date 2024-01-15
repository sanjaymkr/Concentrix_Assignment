import re
import traceback
from datetime import date, datetime

import datefinder
from dateutil import parser
from datetimerange import DateTimeRange

re_flags = re.MULTILINE | re.IGNORECASE
mobile_re = re.compile(r"\b[\+]?(\d{10,13}|[\(][\+]{0,1}\d{2,5}[\)]?\d{8,10}|\d{2,6}[\-]{1}\d{5,13}[\-]?\d{5,13})\b", re_flags)
only_year_pattern = re.compile(r'\d{4}')
common_format = re.compile(r"\d{1,2}[ ]*[/][ ]*\d{1,2}[ ]*[/][ ]*\d{3,4}")
email_re = re.compile(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", re_flags)
symbols_re = re.compile(r"[@,.%!@#$%^&*()_+=\[\]\{\}:;\"'<,>.?]", re_flags)
today_date_re = re.compile(r"\b(present|till now|current|currently|till date|till|onwards)\b", re_flags)
years_re = re.compile(r'(\d+(?:\.\d{1,2})?\+*[\s]*years|\d+(?:\.\d{1,2})?\+*[\s]*months|\d+(?:\.\d{1,2})?\+*[\s]*yrs)', re_flags)
hypen_space_re = re.compile(r"\b(\- )\b|\b( \-)\b", re_flags)
unwanted_date_element = re.compile(r"T\d{2}\:\d{2}\:\d{2}\b", re_flags)
date_re = re.compile(r"(?P<start>(?:\d{1,2}/)?\d{4}(?:\s+[a-z]+)|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,-]?\d{0,2}[\s,'-]?\d{4}|\d{1,2}[a-z]{3}\s\w+\s\d{4}|\w{3}\.\s\d{1,2},\s\d{4}|\d{1,2}\s[A-Za-z]+\s\d{4}|\d{1,2}[-'\s][A-Za-z]{3}[-'\s]\d{2}|\d{1,2}\s[A-Za-z]{3}\b|[a-z]+[-'/,]?\d{1,2}[ '-/,]+\d{1,4}|\d{1,2}?\d{1,2}\d{2,4}|(?:\d{1,2}[/' -]+)?\d{1,2}[ '/-]+\d{3,4}|\d{1,4}/\d{1,4}/\d{1,4}|\d{1,2}'[ ,/'-]*[A-z]+[' ,-/]*\d{1,4}|[A-z]+['/ ,-]*\d{1,2}[' /,-]*[A-z]+[' ,/-]*\d{1,4}|[a-z]+[ -]+\d{1,4}|[A-z]+[ /',-]*\d{1,2}[' ,-/]*[A-z]+[ ,'-/]*\d{1,4}|[a-z][ -/']+\d+[ -'/]\d{1,4}|\|\d{1,2}-\d{1,2]-\d{2,4}|\d+[-,' /]+\d{1,4}|(?:\d{1,2}|[a-zA-Z]{3})[ ,](?:\d{1,2}|[a-zA-Z]{3})[ ,]\d{1,4}|\d+[a-z]+[ -/'][a-z]+[-' /]\d{1,4}|[0-9]{1,2}[- ][A-Za-z]{3}[- ]\d{4}|[0-9]{1,2}[- ][A-Za-z]+[- ,]+\d{4}|[a-z]+[ -]+\d{1,4}|[A-Za-z]{3}\.?\s\d{1,2},?\s\d{4}|[0-9]{1,2}\/[0-9]{1,2}\/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2,4})?(?:)", re_flags)
total_experience_re = re.compile(r"(?P<year>[\d]?[.]?\d+[ +]*)(years?|yrs)[ ,*]?(and)?[, ]?((?P<month>[\d]+[ ,])(months?))?", re_flags)
digit_start_pattern = re.compile(pattern=r"^\d+", flags=re_flags)
to_regex_in_date_re = re.compile(r"\bto\b", re_flags)
unwanted_date_element_new = re.compile(r"\bfrom\b|\bsince\b", re_flags)
desired_date_format_pattern = re.compile(r'^\d{2}-\d{2}-\d{4}')
month_year_date_format_regex = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z,\s]*\s?\d{1,2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\d{1,2}\b', re.IGNORECASE)

def fn_convert_yymmdd_to_ddmmy(text:str):
    try:
        if text is not None:
            converted_date = datetime.strftime(datetime.strptime(str(text), '%Y-%m-%d'), '%d-%m-%Y')
            return converted_date
    except:
        return text

def starting_date_repl(text:str):
    starting_dates = digit_start_pattern.findall(string=text)
    if len(starting_dates) == 0:
        text = "01 " + text
    return text

def preprocess_date_sublists(date):
    try:
        if re.match(today_date_re,date):
            date = date.today().strftime('%d-%m-%Y')
        elif re.match(common_format,date):
            try:
                obj = datetime.strptime(date, "%d / %m / %Y")
                date = obj.strftime("%d-%m-%Y")
            except:
                obj = datetime.strptime(date, "%d/%m/%Y")
                date = obj.strftime("%d-%m-%Y")
        elif re.match(r'\d{2}-\d{2}-\d{4}',date):
            pass
        else:
            matches = datefinder.find_dates(date)
            for match in matches:
                date = match.strftime('%d-%m-%Y')
    except Exception as e:
        e = traceback.format_exc()

    return date

def preprocess_only_years(item):
    try:
        if re.match(r'\d{2}-\d{2}-\d{4}', item):
            return item
        else:
            only_year = ''.join(only_year_pattern.findall(item))
            if only_year:
                return date.today().replace(year=int(only_year)).strftime('%d-%m-%Y')
    except Exception as e:
        e = traceback.format_exc()

def month_year_date_format_handling(ip:list):
    try:
        final_date_output = []

        for str_of_date in ip:
            if bool(desired_date_format_pattern.match(str_of_date))==True:
                final_date_output.append(str_of_date)

            elif bool(desired_date_format_pattern.match(str_of_date))==False:
                final_date_output.extend(fetch_mmm_yy_format(str_of_date)) 

        formatted_dates = []

        for i in final_date_output:
            date = parser.parse(i, dayfirst=True)
            formatted_dates.append(date.strftime("%d-%m-%Y"))

        return formatted_dates

    except Exception as e:
        return ip
    
def fetch_mmm_yy_format(str_of_date:str):
    try:
        date_output = []

        matches = re.findall(month_year_date_format_regex, str_of_date) if (len(str_of_date)>=4 & str_of_date.strip()[-4:].isdigit()==False) else []

        for i in matches:
            month_pattern = re.compile(r"[a-zA-Z]*")
            month_matches = month_pattern.findall(i)
            month = [i[0:3] for i in month_matches if i!='']
            month = month[0] if month != [] else ''
            
            year_pattern = re.compile(r"[0-9]{1,2}")
            year_matches = year_pattern.findall(i)
            year = [str(2000+ int(i)) for i in year_matches if i!='']
            year = year[0] if year != [] and int(year[0])<= datetime.today().date().year else ''

            date_output.append(month + " " + year)

        return date_output if matches not in [[],None] else date_output

    except Exception as e:
        return str_of_date

def fn_find_dates(text: str):
    found_dates = []
    try:
        today = date.today()
        d1 = " " + today.strftime("%d-%m-%Y")

        r = today_date_re.sub(repl=d1, string=text)
        r = to_regex_in_date_re.sub(repl=" - ", string=r)
        r = symbols_re.sub(repl=" ", string=r)
        r = mobile_re.sub(repl=" ", string=r)
        r = email_re.sub(repl=" ", string=r)
        r = years_re.sub(repl=" ",string=r)
        r=  unwanted_date_element_new.sub(repl=" ",string=r)
        r = hypen_space_re.sub(repl=" - ", string=r)
        r_list = r.split("\n")
        found_dates = []

        if len(found_dates) == 0:
            date_sublists = [match.group('start') for match in date_re.finditer(" ".join(r_list)) if match.group('start')]
            date_sublists.extend([match.group('end') for match in date_re.finditer(" ".join(r_list)) if match.group('end')])
            if len(date_sublists)==2:
                if ''.join(only_year_pattern.findall(date_sublists[0])):
                    if date_sublists[0].isdigit():
                        year = int(date_sublists[0])
                        new_date = datetime(year, 1, 1).strftime('%d-%m-%Y')
                        date_sublists[0] = new_date
            date_sublists = list(map(preprocess_date_sublists,date_sublists))
            date_sublists = month_year_date_format_handling(date_sublists)
            found_dates = list(map(preprocess_only_years,date_sublists))

        if len(found_dates) == 0:
            for i in r_list:
                try:
                    time_range = DateTimeRange.from_range_text(i)
                    time_range = unwanted_date_element.sub(repl="", string=str(time_range))
                    time_range = str(time_range).split(" - ")
                    found_dates.extend(time_range)
                except:
                    pass

        if len(found_dates) == 0:
            try:
                matches = datefinder.find_dates("".join(r_list))
                found_dates = [match.strftime("%d-%m-%Y") for match in matches]
            except:
                pass

        found_dates = list(map(fn_convert_yymmdd_to_ddmmy, found_dates))

    except Exception as e:
        pass

    return list(filter(lambda x: x!= None,found_dates))

def get_pairs_of_dates(date_list = []):
    date_list = list(map(lambda x : datetime.strptime(x, "%d-%m-%Y"), date_list))
    date_list.sort()
    if len(date_list)%2 != 0:
        date_list.append(datetime.today())
    return date_list

def get_exp_bw_two_dates(dt1, dt2):
    exp = 0
    delta = dt2 - dt1
    yrs = delta.days / 365
    exp = exp + yrs
    return exp

def get_exp_from_pairs_of_dates(date_list = []):
    date_list = list(map(lambda x : datetime.strptime(x, "%d-%m-%Y"), date_list))
    date_list.sort()
    if len(date_list)%2 != 0:
        date_list.append(datetime.today())
    date_pairs_list = [[date_list[i],date_list[i+1]] for i in range(0, len(date_list), 2)]
    date_pairs_exp_calc = list(map(lambda x: get_exp_bw_two_dates(x[0], x[1]), date_pairs_list))
    return sum(date_pairs_exp_calc)

def fn_calculate_exp(dt: list):
    exp = 0
    try:
        if dt is not None and len(dt) > 0:
            exp = 0
            for i in dt:
                i_datetime = list(map(lambda x : datetime.strptime(x, "%d-%m-%Y"), i))
                if i is not None and len(i) >= 1:
                    exp_yrs = get_exp_from_pairs_of_dates(i)
                    exp = exp + exp_yrs

            exp = round(exp, 2)
            exp = abs(exp)

    except Exception as e:
        pass

    return exp

def total_experience_from_text(text):
    text = text.replace('+','')
    try:
        for m in total_experience_re.finditer(text):
            years = float(m.group('year'))*12
            month = m.group('month')
            if month:
                month = float(month)
                total = years + month
                total /= 12
                return round(total,2)
            else:
                total = years / 12
                return round(total,2)
    except Exception as e:
        pass
