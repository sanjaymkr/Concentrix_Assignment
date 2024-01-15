from pathlib import Path
import glob
from ml_parser_api.libs.concentrix_ml_parser.parsers.resume_parser_main import (fn_main)

# cv_result = fn_main(r"/home/marktine/Documents/JD bugs/Tejas_AI_Engineer_Infowind_P (1).docx")
# print(cv_result)

import pandas as pd
# Create an empty list to store the results
results = []

# Iterate over the files in the directory
for path in glob.glob("/home/marktine/Downloads/A_for_test/*"):
    try:
        result = fn_main(path)
        results.append(result)
    except Exception as e:
        print(f"Error processing file {path}: {str(e)}")
        continue  # Continue to the next file

# Create a dataframe from the results list
df = pd.DataFrame(results)

# Extract the filename without extension
f_name = os.path.split(path)[-1]
filename = f"{f_name.split('.')[0]}_result.xlsx"

# Export the dataframe to an Excel file
df.to_excel(f"/home/marktine/Documents/concentrix_test_resumelt/{filename}", index=False)

