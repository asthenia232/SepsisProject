import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import plotly.express as px





# Pearson's Correlation for sepsis vs no sepsis re: health metric
def pearson_correlation_categorical(csv_file, health_parameter):

   df = pd.read_csv(csv_file)

   sepsis_group = df[df['sepsislabel'] == 1][health_parameter]
   no_sepsis_group = df[df['sepsislabel'] == 0][health_parameter]


   correlation_coefficient, p_value = pearsonr(sepsis_group, no_sepsis_group)

   print("Pearson correlation coefficient:", correlation_coefficient)
   print("P-value", p_value)





# Pearsons correlation for postadmission and 6H before in sepsis group for health metric
def pearson_correlation_longitudinal(csv_file, health_parameter):

   df = pd.read_csv(csv_file)

   six_hour_before = df[f'{column}_six_hour_before_admission']
   postadmission = df[f'{column}_postadmission_hour0_sepsis']


   correlation_coefficient, p_value = pearsonr(six_hour_before, postadmission)

   print("Pearson correlation coefficient:", correlation_coefficient)
   print("P-value", p_value)


# defaulted to temporal csv due to only normal data between those variables
def pearsons_coefficient(csv_file, column):
    df = pd.read_csv(csv_file)

    six_hour_before = df[f'{column}_six_hour_before_admission']
    postadmission = df[f'{column}_postadmission_hour0_sepsis']

    paired_data = pd.concat([six_hour_before, postadmission], axis=1,
                keys=['six_hour_before', 'postadmission']).dropna()

    correlation_coefficient, p_value = pearsonr(paired_data['six_hour_before'], paired_data['postadmission'])

    print(f"Pearson's Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")


def spearmans_coefficient_temporal(csv_file, column):
    df = pd.read_csv(csv_file)

    six_hour_before = df[f'{column}_six_hour_before_admission']
    postadmission = df[f'{column}_postadmission_hour0_sepsis']

    paired_data = pd.concat([six_hour_before, postadmission], axis=1,
                keys=['six_hour_before', 'postadmission']).dropna()

    correlation_coefficient, p_value = spearmanr(paired_data['six_hour_before'], paired_data['postadmission'])

    print(f"Spearman's Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")


def spearmans_coefficient_sepsis_vs_no_sepsis(column):
    df = pd.read_csv('avg_numeric_grouped_by_id+sepsislabel.csv')

    # Filter data
    sepsis_0_data = df[df['sepsislabel'] == 0]
    sepsis_1_data = df[df['sepsislabel'] == 1]

    paired_data = pd.concat([sepsis_0_data, sepsis_1_data], axis=1,
                keys=['sepsis_0_data', 'sepsis_1_data']).dropna()

    correlation_coefficient, p_value = spearmanr(paired_data['sepsis_0_data'], paired_data['sepsis_1_data'])

    print(f"Spearman's Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")


pearsons_coefficient('temporal_non_vitals.csv', 'ph')



# avgs divided by patient
df = pd.read_csv('avg_numeric_grouped_by_id+sepsislabel.csv')
corr_matrix = df.corr()
fig_rdbu = px.imshow(corr_matrix, color_continuous_scale='RdBu')
fig_rdbu.show()



df = pd.read_csv('temporal_non_vitals.csv')
corr_matrix = df.corr()
fig_rdbu = px.imshow(corr_matrix, color_continuous_scale='RdBu')
fig_rdbu.show()


