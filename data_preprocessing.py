import csv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from connect import connect
from config import config
from decimal import Decimal


filename= 'data/dataset.csv'
df = pd.read_csv('data/dataset.csv')

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#
print(df.shape)

# Check the headers of columns
print(df.columns)

# Cursory glance at snippet of table
print(df.head(15))


# -------------------------------- Cleaning and exploratory analysis through pandas with psycopg2 integration ----------------------------------
# pandas with psycopg2 integration


# function to execute sql query and save as csv if none exist
def export_query_to_csv(query, csv_file_path):

    #checks to see if file already present before committing function
    if not os.path.exists(csv_file_path):
        conn = psycopg2.connect(database="sepsis", user='postgres', password='*******', host='localhost',
                                port='5432')
        cur = conn.cursor()
        # Generate the COPY command with the query
        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)

        with open(csv_file_path, 'w', newline='') as csv_file:
            cur.copy_expert(outputquery, csv_file)

        cur.close()
        conn.close()

        print("file saved successfully.")
    else:
        print("file already found.")



# creating function for query visualisations
def query_visualization_bar(filename, xl, yl, title, columns_to_clean=None):

    # Dictionary storing values from clean_column
    data = {}

    # opens csv and iterates through each row
    with open(filename, 'r') as csv_file:

        # using dictionary key-pairs to make data cleaning more streamlined
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            for column, value in row.items():
                if column != "":
                    clean_column = column
                    if columns_to_clean:
                        for text_to_remove in columns_to_clean:

                            # Removing text in column header, specified when calling function.
                            clean_column = clean_column.replace(text_to_remove, '')

                    # Check if the value is an int
                    if value.isdigit():
                        data[clean_column] = int(value)
                    else:
                        # Try converting to float, use as-is if conversion fails
                        try:
                            data[clean_column] = float(value)
                        except ValueError:
                            data[clean_column] = value

    # Extracts keys, values from cleaned dictionary
    headers = list(data.keys())
    values = list(data.values())


    # Matplotlib bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(headers, values, color='blue')
    plt.xlabel(xl) # variable for x-axis label
    plt.ylabel(yl) # variable for y-axis label
    plt.title(title) # variable for title
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



# Query to extract percentage of nulls

query = """SELECT
            ROUND(
                (SUM(CASE WHEN Hour IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Hour_missing_percentage,
            ROUND(
                (SUM(CASE WHEN HR IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS HR_missing_percentage,
            ROUND(
                (SUM(CASE WHEN O2Sat IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS O2Sat_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Temp IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Temp_missing_percentage,
            ROUND(
                (SUM(CASE WHEN SBP IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS SBP_missing_percentage,
            ROUND(
                (SUM(CASE WHEN MAP IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS MAP_missing_percentage,
            ROUND(
                (SUM(CASE WHEN DBP IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS DBP_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Resp IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Resp_missing_percentage,
            ROUND(
                (SUM(CASE WHEN EtCO2 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS EtCO2_missing_percentage,
            ROUND(
                (SUM(CASE WHEN BaseExcess IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS BaseExcess_missing_percentage,
            ROUND(
                (SUM(CASE WHEN HCO3 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS HCO3_missing_percentage,
            ROUND(
                (SUM(CASE WHEN FiO2 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS FiO2_missing_percentage,
            ROUND(
                (SUM(CASE WHEN pH IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS pH_missing_percentage,
            ROUND(
                (SUM(CASE WHEN PaCO2 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS PaCO2_missing_percentage,
            ROUND(
                (SUM(CASE WHEN SaO2 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS SaO2_missing_percentage,
            ROUND(
                (SUM(CASE WHEN AST IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS AST_missing_percentage,
            ROUND(
                (SUM(CASE WHEN BUN IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS BUN_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Alkalinephos IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Alkalinephos_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Calcium IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Calcium_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Chloride IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Chloride_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Creatinine IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Creatinine_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Bilirubin_direct IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Bilirubin_direct_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Glucose IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Glucose_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Lactate IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Lactate_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Magnesium IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Magnesium_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Phosphate IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Phosphate_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Potassium IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Potassium_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Bilirubin_total IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Bilirubin_total_missing_percentage,
            ROUND(
                (SUM(CASE WHEN TroponinI IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS TroponinI_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Hct IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Hct_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Hgb IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Hgb_missing_percentage,
            ROUND(
                (SUM(CASE WHEN PTT IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS PTT_missing_percentage,
            ROUND(
                (SUM(CASE WHEN WBC IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS WBC_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Fibrinogen IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Fibrinogen_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Platelets IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Platelets_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Age_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Gender IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Gender_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Unit1 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Unit1_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Unit2 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Unit2_missing_percentage,
            ROUND(
                (SUM(CASE WHEN HospAdmTime IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS HospAdmTime_missing_percentage,
            ROUND(
                (SUM(CASE WHEN ICULOS IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS ICULOS_missing_percentage,
            ROUND(
                (SUM(CASE WHEN SepsisLabel IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS SepsisLabel_missing_percentage,
            ROUND(
                (SUM(CASE WHEN Patient_ID IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)::float)::numeric, 2
            ) AS Patient_ID_missing_percentage
        FROM public.patient_data"""

csv_file_path = 'null_values.csv'

export_query_to_csv(query, csv_file_path)



# Visualisation: percentage of null values in bar graph

filename = 'null_values.csv'
xl = 'Columns'
yl = 'Percentage'
title = 'Percent of missing values in table'
columns_to_clean = ['_missing_percentage']
query_visualization_bar(filename, xl, yl, title, columns_to_clean)











# Query to extract the total count of Sepsis vs no Sepsis patients

query = """WITH MaxSepsisPerPatient AS (
    SELECT
        Patient_ID,
        MAX(SepsisLabel) AS MaxSepsis
    FROM public.patient_data
    GROUP BY Patient_ID
)


SELECT
    SUM(CASE WHEN MaxSepsis = 1 THEN 1 ELSE 0 END) AS total_sepsis_patients,
    SUM(CASE WHEN MaxSepsis = 0 THEN 1 ELSE 0 END) AS total_non_sepsis_patients
FROM MaxSepsisPerPatient"""

csv_file_path = 'sepsis_count.csv'

export_query_to_csv(query, csv_file_path)


# Visualisation: count of patients with sepsis and without in dataset

filename = 'sepsis_count.csv'
xl = 'Column'
yl = 'Total patients'
title = 'Total patients grouped by sepsis status'
columns_to_clean = ['_']
query_visualization_bar(filename, xl, yl, title)












# Extracting Sepsis vs no sepsis and grouping by gender


query = ("""WITH maxsepsisperpatient AS (
    SELECT
        patient_ID, gender,
        MAX(SepsisLabel) AS maxSepsis
    FROM public.patient_data
    GROUP BY patient_ID, gender
)

SELECT
	SUM(CASE WHEN maxsepsis = 1 and gender = 0 THEN 1 ELSE 0 END) as female_sepsis,
	SUM(CASE WHEN maxsepsis = 1 and gender = 1 THEN 1 ELSE 0 END) as male_sepsis,
	SUM(CASE WHEN maxsepsis = 0 AND gender = 0 THEN 1 ELSE 0 END) as female_no_sepsis,
	SUM(CASE WHEN maxsepsis = 0 AND gender = 1 THEN 1 ELSE 0 END) as male_no_sepsis
FROM maxsepsisperpatient""")
csv_file_path = 'sepsis_gender.csv'

export_query_to_csv(query, csv_file_path)

# Data better visualised in stacked bar graphs

# Below function for stacked graphs
def query_visualization_stacked_bar(filename, xl, yl, title, columns_to_clean=None):
    data = {}

    # Opening csv and iterating through each row with header columns as dictionary key
    with open(filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            for column, value in row.items():
                label = "_".join(column.split("_")[1:])
                clean_column = label
                if columns_to_clean:
                    for text_to_remove in columns_to_clean:
                        clean_column = clean_column.replace(text_to_remove, '')

                if label not in data:
                    data[label] = []

                # function modified to adjust for NaN and differences in value formats
                if value.isdigit():
                    data[label].append(int(value))
                else:
                    try:
                        data[label].append(float(value))
                    except ValueError:
                        data[label].append(value)

    categories = list(data.keys())
    plt.figure(figsize=(12, 6))

    bottom = None
    for label in data.keys():
        values = data[label]
        plt.bar(categories, values, label=label.title(), bottom=bottom)
        if bottom is None:
            bottom = values
        else:
            bottom = [sum(x) for x in zip(bottom, values)]

    plt.xlabel(xl)  # variable for x-axis label
    plt.ylabel(yl)  # variable for y-axis label
    plt.title(title)  # variable for titl
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


#Visualising split of gender through stacked bar function
filename = 'sepsis_gender.csv'
xl = 'Column'
yl = 'Total patients'
title = 'Total patients grouped by gender'
columns_to_clean = ['_']
query_visualization_bar(filename, xl, yl, title)

# Limitations with labeling based on csv data
# using mathplotlib to manually graph sepsis_gender.csv in stacked bar as only 4 values present
gender = (
    "male",
    "female"
)
sepsis_type = {
    "sepsis": np.array([1739, 1193]),
    "no sepsis": np.array([20827, 16577]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(2)

for boolean, sepsis_type in sepsis_type.items():
    p = ax.bar(gender, sepsis_type, width, label=boolean, bottom=bottom)
    bottom += sepsis_type

ax.set_title("Split of gender and sepsis status")
ax.legend(loc="upper right")

plt.show()











# Extracting Post vs preadmission sepsis from dataset
query = ("""WITH cte AS (
	SELECT *,
     	LAG(sepsislabel) OVER (PARTITION BY patient_id ORDER BY iculos) AS prev_sepsislabel
    FROM public.patient_data
)

SELECT
	COUNT(DISTINCT patient_id) as total_patients,
	COUNT(DISTINCT CASE WHEN sepsislabel = 1 THEN patient_id ELSE NULL END) as total_sepsis_patients,
    COUNT(DISTINCT CASE WHEN hour = 0 AND sepsislabel = 1 THEN patient_id ELSE NULL END) as preadmission_sepsis,
    COUNT(DISTINCT CASE WHEN hour > 0 AND prev_sepsislabel = 0 AND sepsislabel = 1 THEN patient_id ELSE NULL END) as postadmission_sepsis
FROM cte""")

csv_file_path = 'post_pre_admission.csv'

export_query_to_csv(query, csv_file_path)



# Visualisation: split of Total patients and total sepsis with pre and postadmission
filename = 'post_pre_admission.csv'
xl = 'Sepsis Type'
yl = 'Count of Patients'
title = 'Count of Patients Post vs Pre admission to ICU '
query_visualization_bar(filename, xl, yl, title)













#Extracting split of gender between pre vs post admission in those with sepsis

query = """WITH postadmission_sepsis AS (
	SELECT *,
     	LAG(sepsislabel) OVER (PARTITION BY patient_id ORDER BY iculos) AS prev_sepsislabel
    FROM public.patient_data
)

SELECT
	COUNT(DISTINCT CASE WHEN hour = 0 AND sepsislabel = 1 AND gender = 0 THEN patient_id ELSE NULL END) as female_preadmission_sepsis,
	COUNT(DISTINCT CASE WHEN hour = 0 AND sepsislabel = 1 AND gender = 1 THEN patient_id ELSE NULL END) as male_preadmission_sepsis,
    COUNT(DISTINCT CASE WHEN hour > 0 AND prev_sepsislabel = 0 AND sepsislabel = 1 AND gender = 0 THEN patient_id ELSE NULL END) as female_postadmission_sepsis,
	COUNT(DISTINCT CASE WHEN hour > 0 AND prev_sepsislabel = 0 AND sepsislabel = 1 AND gender = 1 THEN patient_id ELSE NULL END) as male_postadmission_sepsis
FROM postadmission_sepsis"""

csv_file_path = 'sepsis_type_gender.csv'

export_query_to_csv(query, csv_file_path)


# Exploratory look at stacked_bar but issues with labelling
query_visualization_stacked_bar('sepsis_type_gender.csv', 'sepsis type', 'Count of patients', 'Split of sepsis admission by gender')


#Visualisation:  Using matplotlib and manually inputted array for cleaner visualisation
gender = (
    "male",
    "female"
)
sepsis_type = {
    "preadmission": np.array([251, 175]),
    "post-admission": np.array([1488, 1018]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(2)

for boolean, sepsis_type in sepsis_type.items():
    p = ax.bar(gender, sepsis_type, width, label=boolean, bottom=bottom)
    bottom += sepsis_type

ax.set_title("Split of gender and sepsis admission")
ax.legend(loc="upper right")

plt.show()














# Extracting avg vital signs for Non-sepsis, preadmission sepsis, postadmission sepsis and 6 hours prior to postadmission

query = ("""WITH cte AS (
    SELECT *,
           LAG(sepsislabel) OVER (PARTITION BY patient_id ORDER BY iculos) AS prev_sepsislabel,
	       LAG(hr, 6) OVER (PARTITION BY patient_id ORDER BY iculos) as six_r_before_sepsishr,
		   LAG(o2sat, 6) OVER (PARTITION BY patient_id ORDER BY iculos) as six_r_before_sepsis_o2,
           LAG(temp, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_temp,
           LAG(sbp, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_sbp,
           LAG(map, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_map,
           LAG(dbp, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_dbp,
           LAG(resp, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_resp,
           LAG(etco2, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_etco2
    FROM public.patient_data
)

SELECT
	avg(CASE WHEN sepsislabel = 0 THEN hr ELSE NULL END) AS avg_hr_no_sepsis,
	avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN hr ELSE NULL END) AS avg_hr_preadmission_sepsis,
	avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN hr ELSE NULL END) AS avg_hr_postadmission_hour0_sepsis,
	avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0  AND six_r_before_sepsishr IS NOT NULL THEN six_r_before_sepsishr ELSE NULL END) as avg_hr_six_hour_before_admission,
	
	avg(CASE WHEN sepsislabel = 0 THEN hr ELSE NULL END) AS avg_o2_no_sepsis,
	avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN hr ELSE NULL END) AS avg_o2_preadmission_sepsis,
	avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN hr ELSE NULL END) AS avg_o2_postadmission_hour0_sepsis,
	avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_o2 IS NOT NULL THEN six_r_before_sepsis_o2 ELSE NULL END) as avg_o2_six_hour_before_admission,

    avg(CASE WHEN sepsislabel = 0 THEN temp ELSE NULL END) AS avg_temp_no_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN temp ELSE NULL END) AS avg_temp_preadmission_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN temp ELSE NULL END) AS avg_temp_postadmission_hour0_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_temp IS NOT NULL THEN six_r_before_sepsis_temp ELSE NULL END) AS avg_temp_six_hour_before_admission,
    
    avg(CASE WHEN sepsislabel = 0 THEN sbp ELSE NULL END) AS avg_sbp_no_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN sbp ELSE NULL END) AS avg_sbp_preadmission_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN sbp ELSE NULL END) AS avg_sbp_postadmission_hour0_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_sbp IS NOT NULL THEN six_r_before_sepsis_sbp ELSE NULL END) AS avg_sbp_six_hour_before_admission,
    
    avg(CASE WHEN sepsislabel = 0 THEN map ELSE NULL END) AS avg_map_no_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN map ELSE NULL END) AS avg_map_preadmission_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN map ELSE NULL END) AS avg_map_postadmission_hour0_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_map IS NOT NULL THEN six_r_before_sepsis_map ELSE NULL END) AS avg_map_six_hour_before_admission,
    
    avg(CASE WHEN sepsislabel = 0 THEN dbp ELSE NULL END) AS avg_dbp_no_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN dbp ELSE NULL END) AS avg_dbp_preadmission_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN dbp ELSE NULL END) AS avg_dbp_postadmission_hour0_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_dbp IS NOT NULL THEN six_r_before_sepsis_dbp ELSE NULL END) AS avg_dbp_six_hour_before_admission,
    
    avg(CASE WHEN sepsislabel = 0 THEN resp ELSE NULL END) AS avg_resp_no_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN resp ELSE NULL END) AS avg_resp_preadmission_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN resp ELSE NULL END) AS avg_resp_postadmission_hour0_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_resp IS NOT NULL THEN six_r_before_sepsis_resp ELSE NULL END) AS avg_resp_six_hour_before_admission,
    
    avg(CASE WHEN sepsislabel = 0 THEN etco2 ELSE NULL END) AS avg_etco2_no_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour = 0 THEN etco2 ELSE NULL END) AS avg_etco2_preadmission_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN etco2 ELSE NULL END) AS avg_etco2_postadmission_hour0_sepsis,
    avg(CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_etco2 IS NOT NULL THEN six_r_before_sepsis_etco2 ELSE NULL END) AS avg_etco2_six_hour_before_admission
FROM cte""")

csv_file_path = 'avg_distribution_by_sepsis.csv'


export_query_to_csv(query, csv_file_path)



# Visualisation: Using bar graph function to get initial visual
filename = 'avg_distribution_by_sepsis.csv'
xl = 'Features'
yl = 'Average Values'
title = 'Average Values Grouped by Features'
query_visualization_bar(filename, xl, yl, title)


# Visualisation: Manually create grouped/clustered column to better represent the data

vital_sign_avg = ("HR (bpm)", "O2 (% Sat)", "Temp (C)", "SBP (mmHg)", "MAP (mmHg)", "DBP (mmHg)", "Resp (BPM)", "End tidal CO2 (mmHg)")
sepsis_admission_type = {
    'No Sepsis' : (84.46533378,	84.46533378,	36.97220161,	123.7923586,	82.44237528,	63.86445418,	18.69435702,	32.98636505),
    'Pre admission Sepsis' : (90.74107143,	90.74107143,	36.53595745,	125.922619,	83.86425926,63.39655172,	18.45283019,	33),
    'Post admission Sepsis': (89.96804836,	89.96804836,	37.18534106,	123.1539727,	81.38298639,	62.6493135,	20.40117349,	32.64644351),
    '6 Hours Before Post admission Sepsis' : (89.21358025,	97.05597579,	37.10403266,	122.9660011,	81.06983944,	62.14283419,	20.41097689,	33.19095477),
}

x = np.arange(len(vital_sign_avg))
width = 0.10
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in sepsis_admission_type.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, labels=[f'{val:.2f}' for val in measurement], padding=3, rotation=90)
    multiplier += 1

ax.set_ylabel('Unit of measurement (*)')
ax.set_title('Avg of vital signs grouped via admission type', loc='left')
ax.set_xticks(x + width * (len(sepsis_admission_type) - 1) / 2, vital_sign_avg)
ax.legend(loc='best', ncols=8)
ax.set_ylim(10, 130)

plt.show()



























