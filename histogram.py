import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import export_query_to_csv

# Histogram function for looking at avg vital signs and laboratory features for sepsis vs no-sepsis
def histogram_visualisation_one_feature(column):
    df = pd.read_csv('avg_numeric_grouped_by_id+sepsislabel.csv')

    # Filter and plot data using Seaborn/Matplotlib
    sepsis_0_data = df[df['sepsislabel'] == 0]
    sepsis_1_data = df[df['sepsislabel'] == 1]

    plt.figure(figsize=(8, 6))

    # Using seaborn for cleaner visual + KDE.
    sns.histplot(sepsis_0_data[column], bins=20, kde=True, color='blue', label='No Sepsis', stat='percent')
    sns.histplot(sepsis_1_data[column], bins=20, kde=True, color='orange', label='Sepsis', stat='percent')
    plt.xlabel(column)
    plt.ylabel('Percent')  # Percent used due to better visualise due to disparity in group sizes
    plt.title(f'Histogram for {column}')
    plt.legend()
    plt.show()



# Histogram function with vital or non-vital csv entered. Longitudinal histogram.
def histogram_visualisation_temporal(csv_file, column):
    df = pd.read_csv(csv_file)

    # Filter via dataframe and plot data using Seaborn/Matplotlib
    six_hr_pre_data = df[f'{column}_six_hour_before_admission']
    post_admission_data = df[f'{column}_postadmission_hour0_sepsis']


    plt.figure(figsize=(10, 8))
    sns.histplot(six_hr_pre_data, bins=20, kde=True, color='blue', label='6 Hours Before Admission')
    sns.histplot(post_admission_data, bins=20, kde=True, color='orange', label='First Hour of PostAdmission')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {column}')
    plt.legend()
    plt.show()








# Extracting avg vital and laboratory values for each patient based on sepsislabel
query = ("""SELECT
    patient_id,
	sepsislabel,
    AVG(HR) AS avg_HR,
    AVG(O2Sat) AS avg_O2Sat,
    AVG(Temp) AS avg_Temp,
    AVG(SBP) AS avg_SBP,
    AVG(MAP) AS avg_MAP,
    AVG(DBP) AS avg_DBP,
    AVG(Resp) AS avg_Resp,
    AVG(EtCO2) AS avg_EtCO2,
    AVG(BaseExcess) AS avg_BaseExcess,
    AVG(HCO3) AS avg_HCO3,
    AVG(FiO2) AS avg_FiO2,
    AVG(pH) AS avg_pH,
    AVG(PaCO2) AS avg_PaCO2,
    AVG(SaO2) AS avg_SaO2,
    AVG(AST) AS avg_AST,
    AVG(BUN) AS avg_BUN,
    AVG(Alkalinephos) AS avg_Alkalinephos,
    AVG(Calcium) AS avg_Calcium,
    AVG(Chloride) AS avg_Chloride,
    AVG(Creatinine) AS avg_Creatinine,
    AVG(Bilirubin_direct) AS avg_Bilirubin_direct,
    AVG(Glucose) AS avg_Glucose,
    AVG(Lactate) AS avg_Lactate,
    AVG(Magnesium) AS avg_Magnesium,
    AVG(Phosphate) AS avg_Phosphate,
    AVG(Potassium) AS avg_Potassium,
    AVG(Bilirubin_total) AS avg_Bilirubin_total,
    AVG(TroponinI) AS avg_TroponinI,
    AVG(Hct) AS avg_Hct,
    AVG(Hgb) AS avg_Hgb,
    AVG(PTT) AS avg_PTT,
    AVG(WBC) AS avg_WBC,
    AVG(Fibrinogen) AS avg_Fibrinogen,
    AVG(Platelets) AS avg_Platelets,
    AVG(Age) AS avg_Age,
    AVG(Gender) AS avg_Gender,
    AVG(ICULOS) AS avg_ICULOS

FROM public.patient_data
GROUP BY patient_id, sepsislabel""")

csv_file_path = 'avg_numeric_grouped_by_id+sepsislabel.csv'

export_query_to_csv(query, csv_file_path)











# Extracting laboratory and categorical values from patients with sepsis at time of postadmission and 6 hours before, filtering out null values.
query = ("""WITH cte AS (
    SELECT *,
           LAG(sepsislabel) OVER (PARTITION BY patient_id ORDER BY iculos) AS prev_sepsislabel,
	       LAG(BaseExcess, 6) OVER (PARTITION BY patient_id ORDER BY iculos) as six_r_before_sepsis_BaseExcess,
		   LAG(HCO3, 6) OVER (PARTITION BY patient_id ORDER BY iculos) as six_r_before_sepsis_HCO3,
           LAG(FiO2, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_FiO2,
           LAG(pH, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_pH,
           LAG(PaCO2, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_PaCO2,
           LAG(SaO2, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_SaO2,
           LAG(AST, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_AST,
           LAG(BUN, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_BUN,
           LAG(Alkalinephos, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Alkalinephos,
           LAG(Calcium, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Calcium,
           LAG(Chloride, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Chloride,
           LAG(Creatinine, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Creatinine,
           LAG(Bilirubin_direct, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Bilirubin_direct,
           LAG(Glucose, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Glucose,
           LAG(Lactate, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Lactate,
           LAG(Magnesium, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Magnesium,
           LAG(Phosphate, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Phosphate,
           LAG(Potassium, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Potassium,
           LAG(Bilirubin_total, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Bilirubin_total,
           LAG(TroponinI, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_TroponinI,
           LAG(Hct, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Hct,
           LAG(Hgb, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Hgb,
           LAG(PTT, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_PTT,
           LAG(WBC, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_WBC,
           LAG(Fibrinogen, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Fibrinogen,
           LAG(Platelets, 6) OVER (PARTITION BY patient_id ORDER BY iculos) AS six_r_before_sepsis_Platelets
    FROM public.patient_data
)

SELECT
    BaseExcess_postadmission_hour0_sepsis,
    BaseExcess_six_hour_before_admission,
    
    HCO3_postadmission_hour0_sepsis,
    HCO3_six_hour_before_admission,

    FiO2_postadmission_hour0_sepsis,
    FiO2_six_hour_before_admission,

    pH_postadmission_hour0_sepsis,
    pH_six_hour_before_admission,

    PaCO2_postadmission_hour0_sepsis,
    PaCO2_six_hour_before_admission,

    SaO2_postadmission_hour0_sepsis,
    SaO2_six_hour_before_admission,

    AST_postadmission_hour0_sepsis,
    AST_six_hour_before_admission,

    BUN_postadmission_hour0_sepsis,
    BUN_six_hour_before_admission,

    Alkalinephos_postadmission_hour0_sepsis,
    Alkalinephos_six_hour_before_admission,

    Calcium_postadmission_hour0_sepsis,
    Calcium_six_hour_before_admission,

    Chloride_postadmission_hour0_sepsis,
    Chloride_six_hour_before_admission,

    Creatinine_postadmission_hour0_sepsis,
    Creatinine_six_hour_before_admission,

    Bilirubin_direct_postadmission_hour0_sepsis,
    Bilirubin_direct_six_hour_before_admission,

    Glucose_postadmission_hour0_sepsis,
    Glucose_six_hour_before_admission,

    Lactate_postadmission_hour0_sepsis,
    Lactate_six_hour_before_admission,

    Magnesium_postadmission_hour0_sepsis,
    Magnesium_six_hour_before_admission,

    Phosphate_postadmission_hour0_sepsis,
    Phosphate_six_hour_before_admission,

    Potassium_postadmission_hour0_sepsis,
    Potassium_six_hour_before_admission,

    Bilirubin_total_postadmission_hour0_sepsis,
    Bilirubin_total_six_hour_before_admission,

    TroponinI_postadmission_hour0_sepsis,
    TroponinI_six_hour_before_admission,

    Hct_postadmission_hour0_sepsis,
    Hct_six_hour_before_admission,

    Hgb_postadmission_hour0_sepsis,
    Hgb_six_hour_before_admission,

    PTT_postadmission_hour0_sepsis,
    PTT_six_hour_before_admission,

    WBC_postadmission_hour0_sepsis,
    WBC_six_hour_before_admission,

    Fibrinogen_postadmission_hour0_sepsis,
    Fibrinogen_six_hour_before_admission,

    Platelets_postadmission_hour0_sepsis,
    Platelets_six_hour_before_admission
FROM (
    SELECT
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN BaseExcess ELSE NULL END) AS BaseExcess_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_BaseExcess IS NOT NULL THEN six_r_before_sepsis_BaseExcess ELSE NULL END) AS BaseExcess_six_hour_before_admission,
        
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN HCO3 ELSE NULL END) AS HCO3_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_HCO3 IS NOT NULL THEN six_r_before_sepsis_HCO3 ELSE NULL END) AS HCO3_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN FiO2 ELSE NULL END) AS FiO2_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_FiO2 IS NOT NULL THEN six_r_before_sepsis_FiO2 ELSE NULL END) AS FiO2_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN pH ELSE NULL END) AS pH_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_pH IS NOT NULL THEN six_r_before_sepsis_pH ELSE NULL END) AS pH_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN PaCO2 ELSE NULL END) AS PaCO2_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_PaCO2 IS NOT NULL THEN six_r_before_sepsis_PaCO2 ELSE NULL END) AS PaCO2_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN SaO2 ELSE NULL END) AS SaO2_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_SaO2 IS NOT NULL THEN six_r_before_sepsis_SaO2 ELSE NULL END) AS SaO2_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN AST ELSE NULL END) AS AST_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_AST IS NOT NULL THEN six_r_before_sepsis_AST ELSE NULL END) AS AST_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN BUN ELSE NULL END) AS BUN_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_BUN IS NOT NULL THEN six_r_before_sepsis_BUN ELSE NULL END) AS BUN_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Alkalinephos ELSE NULL END) AS Alkalinephos_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Alkalinephos IS NOT NULL THEN six_r_before_sepsis_Alkalinephos ELSE NULL END) AS Alkalinephos_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Calcium ELSE NULL END) AS Calcium_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Calcium IS NOT NULL THEN six_r_before_sepsis_Calcium ELSE NULL END) AS Calcium_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Chloride ELSE NULL END) AS Chloride_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Chloride IS NOT NULL THEN six_r_before_sepsis_Chloride ELSE NULL END) AS Chloride_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Creatinine ELSE NULL END) AS Creatinine_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Creatinine IS NOT NULL THEN six_r_before_sepsis_Creatinine ELSE NULL END) AS Creatinine_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Bilirubin_direct ELSE NULL END) AS Bilirubin_direct_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Bilirubin_direct IS NOT NULL THEN six_r_before_sepsis_Bilirubin_direct ELSE NULL END) AS Bilirubin_direct_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Glucose ELSE NULL END) AS Glucose_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Glucose IS NOT NULL THEN six_r_before_sepsis_Glucose ELSE NULL END) AS Glucose_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Lactate ELSE NULL END) AS Lactate_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Lactate IS NOT NULL THEN six_r_before_sepsis_Lactate ELSE NULL END) AS Lactate_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Magnesium ELSE NULL END) AS Magnesium_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Magnesium IS NOT NULL THEN six_r_before_sepsis_Magnesium ELSE NULL END) AS Magnesium_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Phosphate ELSE NULL END) AS Phosphate_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Phosphate IS NOT NULL THEN six_r_before_sepsis_Phosphate ELSE NULL END) AS Phosphate_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Potassium ELSE NULL END) AS Potassium_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Potassium IS NOT NULL THEN six_r_before_sepsis_Potassium ELSE NULL END) AS Potassium_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Bilirubin_total ELSE NULL END) AS Bilirubin_total_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Bilirubin_total IS NOT NULL THEN six_r_before_sepsis_Bilirubin_total ELSE NULL END) AS Bilirubin_total_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN TroponinI ELSE NULL END) AS TroponinI_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_TroponinI IS NOT NULL THEN six_r_before_sepsis_TroponinI ELSE NULL END) AS TroponinI_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Hct ELSE NULL END) AS Hct_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Hct IS NOT NULL THEN six_r_before_sepsis_Hct ELSE NULL END) AS Hct_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Hgb ELSE NULL END) AS Hgb_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Hgb IS NOT NULL THEN six_r_before_sepsis_Hgb ELSE NULL END) AS Hgb_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN PTT ELSE NULL END) AS PTT_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_PTT IS NOT NULL THEN six_r_before_sepsis_PTT ELSE NULL END) AS PTT_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN WBC ELSE NULL END) AS WBC_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_WBC IS NOT NULL THEN six_r_before_sepsis_WBC ELSE NULL END) AS WBC_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Fibrinogen ELSE NULL END) AS Fibrinogen_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Fibrinogen IS NOT NULL THEN six_r_before_sepsis_Fibrinogen ELSE NULL END) AS Fibrinogen_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN Platelets ELSE NULL END) AS Platelets_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_Platelets IS NOT NULL THEN six_r_before_sepsis_Platelets ELSE NULL END) AS Platelets_six_hour_before_admission
    FROM cte
) AS filtered_data
WHERE 
    (BaseExcess_postadmission_hour0_sepsis IS NOT NULL AND BaseExcess_six_hour_before_admission IS NOT NULL) OR
    (HCO3_postadmission_hour0_sepsis IS NOT NULL AND HCO3_six_hour_before_admission IS NOT NULL) OR
    (FiO2_postadmission_hour0_sepsis IS NOT NULL AND FiO2_six_hour_before_admission IS NOT NULL) OR
    (pH_postadmission_hour0_sepsis IS NOT NULL AND pH_six_hour_before_admission IS NOT NULL) OR
    (PaCO2_postadmission_hour0_sepsis IS NOT NULL AND PaCO2_six_hour_before_admission IS NOT NULL) OR
    (SaO2_postadmission_hour0_sepsis IS NOT NULL AND SaO2_six_hour_before_admission IS NOT NULL) OR
    (AST_postadmission_hour0_sepsis IS NOT NULL AND AST_six_hour_before_admission IS NOT NULL) OR
    (BUN_postadmission_hour0_sepsis IS NOT NULL AND BUN_six_hour_before_admission IS NOT NULL) OR
    (Alkalinephos_postadmission_hour0_sepsis IS NOT NULL AND Alkalinephos_six_hour_before_admission IS NOT NULL) OR
    (Calcium_postadmission_hour0_sepsis IS NOT NULL AND Calcium_six_hour_before_admission IS NOT NULL) OR
    (Chloride_postadmission_hour0_sepsis IS NOT NULL AND Chloride_six_hour_before_admission IS NOT NULL) OR
    (Creatinine_postadmission_hour0_sepsis IS NOT NULL AND Creatinine_six_hour_before_admission IS NOT NULL) OR
    (Bilirubin_direct_postadmission_hour0_sepsis IS NOT NULL AND Bilirubin_direct_six_hour_before_admission IS NOT NULL) OR
    (Glucose_postadmission_hour0_sepsis IS NOT NULL AND Glucose_six_hour_before_admission IS NOT NULL) OR
    (Lactate_postadmission_hour0_sepsis IS NOT NULL AND Lactate_six_hour_before_admission IS NOT NULL) OR
    (Magnesium_postadmission_hour0_sepsis IS NOT NULL AND Magnesium_six_hour_before_admission IS NOT NULL) OR
    (Phosphate_postadmission_hour0_sepsis IS NOT NULL AND Phosphate_six_hour_before_admission IS NOT NULL) OR
    (Potassium_postadmission_hour0_sepsis IS NOT NULL AND Potassium_six_hour_before_admission IS NOT NULL) OR
    (Bilirubin_total_postadmission_hour0_sepsis IS NOT NULL AND Bilirubin_total_six_hour_before_admission IS NOT NULL) OR
    (TroponinI_postadmission_hour0_sepsis IS NOT NULL AND TroponinI_six_hour_before_admission IS NOT NULL) OR
    (Hct_postadmission_hour0_sepsis IS NOT NULL AND Hct_six_hour_before_admission IS NOT NULL) OR
    (Hgb_postadmission_hour0_sepsis IS NOT NULL AND Hgb_six_hour_before_admission IS NOT NULL) OR
    (PTT_postadmission_hour0_sepsis IS NOT NULL AND PTT_six_hour_before_admission IS NOT NULL) OR
    (WBC_postadmission_hour0_sepsis IS NOT NULL AND WBC_six_hour_before_admission IS NOT NULL) OR
    (Fibrinogen_postadmission_hour0_sepsis IS NOT NULL AND Fibrinogen_six_hour_before_admission IS NOT NULL) OR
    (Platelets_postadmission_hour0_sepsis IS NOT NULL AND Platelets_six_hour_before_admission IS NOT NULL)
""")

csv_file_path = 'temporal_non_vitals.csv'

export_query_to_csv(query, csv_file_path)













## Extracting vital sign values from patients with sepsis at time of postadmission and 6 hours before, filtering out null values.

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
    hr_postadmission_hour0_sepsis,
    hr_six_hour_before_admission,
    
    o2_postadmission_hour0_sepsis,
    o2_six_hour_before_admission,

    temp_postadmission_hour0_sepsis,
    temp_six_hour_before_admission,

    sbp_postadmission_hour0_sepsis,
    sbp_six_hour_before_admission,

    map_postadmission_hour0_sepsis,
    map_six_hour_before_admission,

    dbp_postadmission_hour0_sepsis,
    dbp_six_hour_before_admission,

    resp_postadmission_hour0_sepsis,
    resp_six_hour_before_admission,

    etco2_postadmission_hour0_sepsis,
    etco2_six_hour_before_admission
FROM (
    SELECT
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN hr ELSE NULL END) AS hr_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsishr IS NOT NULL THEN six_r_before_sepsishr ELSE NULL END) AS hr_six_hour_before_admission,
        
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN o2sat ELSE NULL END) AS o2_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_o2 IS NOT NULL THEN six_r_before_sepsis_o2 ELSE NULL END) AS o2_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN temp ELSE NULL END) AS temp_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_temp IS NOT NULL THEN six_r_before_sepsis_temp ELSE NULL END) AS temp_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN sbp ELSE NULL END) AS sbp_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_sbp IS NOT NULL THEN six_r_before_sepsis_sbp ELSE NULL END) AS sbp_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN map ELSE NULL END) AS map_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_map IS NOT NULL THEN six_r_before_sepsis_map ELSE NULL END) AS map_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN dbp ELSE NULL END) AS dbp_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_dbp IS NOT NULL THEN six_r_before_sepsis_dbp ELSE NULL END) AS dbp_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN resp ELSE NULL END) AS resp_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_resp IS NOT NULL THEN six_r_before_sepsis_resp ELSE NULL END) AS resp_six_hour_before_admission,

        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 THEN etco2 ELSE NULL END) AS etco2_postadmission_hour0_sepsis,
        (CASE WHEN sepsislabel = 1 AND hour > 0 AND prev_sepsislabel = 0 AND six_r_before_sepsis_etco2 IS NOT NULL THEN six_r_before_sepsis_etco2 ELSE NULL END) AS etco2_six_hour_before_admission
    FROM cte
    WHERE sepsislabel = 1
) subquery
WHERE
    (hr_postadmission_hour0_sepsis IS NOT NULL AND
    hr_six_hour_before_admission IS NOT NULL) OR
    (o2_postadmission_hour0_sepsis IS NOT NULL AND
    o2_six_hour_before_admission IS NOT NULL) OR
    (temp_postadmission_hour0_sepsis IS NOT NULL OR
    temp_six_hour_before_admission IS NOT NULL) OR
    (sbp_postadmission_hour0_sepsis IS NOT NULL AND
    sbp_six_hour_before_admission IS NOT NULL) OR
    (map_postadmission_hour0_sepsis IS NOT NULL AND
    map_six_hour_before_admission IS NOT NULL) OR
    (dbp_postadmission_hour0_sepsis IS NOT NULL AND
    dbp_six_hour_before_admission IS NOT NULL) OR
    (resp_postadmission_hour0_sepsis IS NOT NULL AND
    resp_six_hour_before_admission IS NOT NULL) OR
    (etco2_postadmission_hour0_sepsis IS NOT NULL AND
    etco2_six_hour_before_admission IS NOT NULL)

""")

csv_file_path = 'temporal_vitals.csv'

export_query_to_csv(query, csv_file_path)







# -----------------------   Exploratory look at distributions to determine suitability for Shapiro Wilk normal distribution testing  ----------------------
# Note: unable to parse through as a list into the function, but working as is.

# histogram plots comparing postadmission sepsis health measures vs 6 hours prior

histogram_visualisation_temporal('temporal_vitals.csv', 'hr')
histogram_visualisation_temporal('temporal_vitals.csv', 'o2')
histogram_visualisation_temporal('temporal_vitals.csv', 'temp')
histogram_visualisation_temporal('temporal_vitals.csv', 'sbp')
histogram_visualisation_temporal('temporal_vitals.csv', 'map')
histogram_visualisation_temporal('temporal_vitals.csv', 'dbp')
histogram_visualisation_temporal('temporal_vitals.csv', 'resp')
histogram_visualisation_temporal('temporal_vitals.csv', 'etco2')


histogram_visualisation_temporal('temporal_non_vitals.csv', 'baseexcess')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'hco3')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'fio2')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'ph')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'paco2')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'sao2')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'ast')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'bun')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'alkalinephos')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'calcium')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'chloride')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'creatinine')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'bilirubin_direct')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'glucose')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'lactate')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'magnesium')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'phosphate')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'potassium')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'bilirubin_total')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'troponini')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'hct')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'hgb')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'ptt')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'wbc')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'fibrinogen')
histogram_visualisation_temporal('temporal_non_vitals.csv', 'platelets')


# Histograms comparing sepsis vs no sepsis distribution
histogram_visualisation_one_feature('avg_hr')
histogram_visualisation_one_feature('avg_o2sat')
histogram_visualisation_one_feature('avg_temp')
histogram_visualisation_one_feature('avg_sbp')
histogram_visualisation_one_feature('avg_map')
histogram_visualisation_one_feature('avg_dbp')
histogram_visualisation_one_feature('avg_resp')
histogram_visualisation_one_feature('avg_etco2')
histogram_visualisation_one_feature('avg_baseexcess')
histogram_visualisation_one_feature('avg_hco3')
histogram_visualisation_one_feature('avg_fio2')
histogram_visualisation_one_feature('avg_ph')
histogram_visualisation_one_feature('avg_paco2')
histogram_visualisation_one_feature('avg_sao2')
histogram_visualisation_one_feature('avg_ast')
histogram_visualisation_one_feature('avg_bun')
histogram_visualisation_one_feature('avg_alkalinephos')
histogram_visualisation_one_feature('avg_calcium')
histogram_visualisation_one_feature('avg_chloride')
histogram_visualisation_one_feature('avg_creatinine')
histogram_visualisation_one_feature('avg_bilirubin_direct')
histogram_visualisation_one_feature('avg_glucose')
histogram_visualisation_one_feature('avg_lactate')
histogram_visualisation_one_feature('avg_magnesium')
histogram_visualisation_one_feature('avg_phosphate')
histogram_visualisation_one_feature('avg_potassium')
histogram_visualisation_one_feature('avg_bilirubin_total')
histogram_visualisation_one_feature('avg_troponini')
histogram_visualisation_one_feature('avg_hct')
histogram_visualisation_one_feature('avg_hgb')
histogram_visualisation_one_feature('avg_ptt')
histogram_visualisation_one_feature('avg_wbc')
histogram_visualisation_one_feature('avg_fibrinogen')
histogram_visualisation_one_feature('avg_platelets')
histogram_visualisation_one_feature('avg_age')
histogram_visualisation_one_feature('avg_gender')
histogram_visualisation_one_feature('avg_iculos')






