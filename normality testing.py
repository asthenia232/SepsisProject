import numpy as np
from scipy.stats import shapiro, ttest_ind, ttest_rel, mannwhitneyu, pearsonr, spearmanr, wilcoxon
import matplotlib.pyplot as plt
import pandas as pd
import csv



# Shapiro wilk test function used with longitudinal data
def shapiro_wilk_test_temporal(csv_file, column):
    df = pd.read_csv(csv_file)

    # Extract the specified columns
    six_hour_before = df[f'{column}_six_hour_before_admission']
    postadmission = df[f'{column}_postadmission_hour0_sepsis']

    # Perform the Shapiro-Wilk test for both groups
    stat1, p_value1 = shapiro(six_hour_before.dropna())
    stat2, p_value2 = shapiro(postadmission.dropna())

    # Print the results for 6 hours prior to post-admission
    print(f"Shapiro-Wilk Test for {six_hour_before}:")
    print(f"Test Statistic: {stat1}")
    print(f"P-value: {p_value1}")
    if p_value1 > alpha:
        print("The data follows a normal distribution (fail to reject H0)")
    else:
        print("The data does not follow a normal distribution (reject H0)")

    # Print the results for post-admission
    print(f"Shapiro-Wilk Test for {postadmission}:")
    print(f"Test Statistic: {stat2}")
    print(f"P-value: {p_value2}")
    if p_value2 > alpha:
        print("The data follows a normal distribution (fail to reject H0)")
    else:
        print("The data does not follow a normal distribution (reject H0)")




# Shapiro wilk test function used with independent groups
def shapiro_wilk_test_categorical(csv_file, column):
    df = pd.read_csv(csv_file)

    # Extract the specified columns
    sepsis_0_data = df[df['sepsislabel'] == 0]
    sepsis_1_data = df[df['sepsislabel'] == 1]

    # Perform the Shapiro-Wilk test for both groups
    stat1, p_value1 = shapiro(sepsis_0_data[column].dropna())
    stat2, p_value2 = shapiro(sepsis_1_data[column].dropna())

    # Print the results for non_sepsis group
    print(f"Shapiro-Wilk Test for {sepsis_0_data[column]}:")
    print(f"Test Statistic: {stat1}")
    print(f"P-value: {p_value1}")
    if p_value1 > alpha:
        print("The data follows a normal distribution (fail to reject H0)")
    else:
        print("The data does not follow a normal distribution (reject H0)")

    # Print the results for sepsis group
    print(f"Shapiro-Wilk Test for {sepsis_1_data[column]}:")
    print(f"Test Statistic: {stat2}")
    print(f"P-value: {p_value2}")
    if p_value2 > alpha:
        print("The data follows a normal distribution (fail to reject H0)")
    else:
        print("The data does not follow a normal distribution (reject H0)")






# Independent T-Test used to compare difference between two distinct groups
def independent_samples_ttest(csv_file, health_parameter):
    df = pd.read_csv(csv_file)

    sepsis_group = df[df['sepsislabel'] == 1][health_parameter]
    no_sepsis_group = df[df['sepsislabel'] == 0][health_parameter]

    t_statistic, p_value = ttest_ind(sepsis_group.dropna(), no_sepsis_group.dropna())

    alpha = 0.05
    print("p-value:", p_value)
    print("T-statistic",  t_statistic)

    if p_value < alpha:
        print(
            f'Reject the null hypothesis: There is a significant difference in {column} between sepsis and no sepsis.'
        )
    else:
        print(
            f'Fail to reject the null hypothesis: There is no significant difference in {column} between sepsis and no sepsis.')





# Paired T-Test used for longitudinal difference in data that follows normal distribution
def paired_samples_ttest(csv_file, column):
    df = pd.read_csv(csv_file)

    six_hour_before = df[f'{column}_six_hour_before_admission']
    postadmission = df[f'{column}_postadmission_hour0_sepsis']

    # Drop rows with missing values for both arrays
    paired_data = pd.concat([six_hour_before, postadmission], axis=1,
                            keys=['six_hour_before', 'postadmission']).dropna()

    t_statistic, p_value = ttest_rel(paired_data['six_hour_before'], paired_data['postadmission'])

    alpha = 0.05
    print("p-value:", p_value)
    print("t-statistic:", t_statistic)

    if p_value < alpha:
        print(
            f'Reject the null hypothesis: There is a significant difference in {column} between post-admission and six hours before post-admission.'
        )
    else:
        print(
            f'Fail to reject the null hypothesis: There is no significant difference in {column} between post-admission and six hours before post-admission.')


# Mann-Whitney U test: Used when groups do not follow a normal distribution
def mann_whitney_u_test(csv_file, health_parameter):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Separate data into sepsis and no-sepsis groups based on 'sepsislabel'
    sepsis_group = df[df['sepsislabel'] == 1][health_parameter]
    no_sepsis_group = df[df['sepsislabel'] == 0][health_parameter]

    # Perform Mann-Whitney U test to compare the two groups
    u_statistic, p_value = mannwhitneyu(sepsis_group.dropna(), no_sepsis_group.dropna())

    # Set the significance level (alpha) for the test
    alpha = 0.05

    # Print test results
    print("p-value:", p_value)
    print("u-statistic:", u_statistic)

    # Check if the p-value is less than the significance level
    if p_value < alpha:
        print(f'Reject the null hypothesis: There is a significant difference in {health_parameter} between post-admission and six hours before post-admission.')
    else:
        print(f'Fail to reject the null hypothesis: There is no significant difference in {health_parameter} between post-admission and six hours before post-admission.')



# Welch's Test: Used when one outcome is not null and the other is on continuous data set
def welchs_t_test(csv_file, health_parameter):
    df = pd.read_csv(csv_file)

    # Extract the specified columns
    six_hour_before = df[f'{health_parameter}_six_hour_before_admission']
    postadmission = df[f'{health_parameter}_postadmission_hour0_sepsis']

    # Perform Welch's t-test
    stat, p_value = ttest_ind(six_hour_before.dropna(), postadmission.dropna(), equal_var=False)

    # Print the results
    print(f"Welch's t-test for {health_parameter}:")
    print(f"Test Statistic: {stat}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print("Reject the null hypothesis: The means are significantly different")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different")





# Wilcoxon signed-rank test: Used for non-normal distribution on longitudinal data
def wilcoxon_temporal(csv_file, health_parameter):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the data for the two time points
    six_hour_before = df[f'{health_parameter}_six_hour_before_admission']
    postadmission = df[f'{health_parameter}_postadmission_hour0_sepsis']

    # Perform Wilcoxon signed-rank test to compare paired samples
    w_statistic, p_value = wilcoxon(six_hour_before, postadmission, nan_policy='omit')

    # Set the significance level (alpha) for the test
    alpha = 0.05

    # Print test results
    print(f"Wilcoxon's signed-rank test for {health_parameter}:")
    print("p-value:", p_value)
    print("w-statistic:", w_statistic)

    # Check if the p-value is less than the significance level
    if p_value < alpha:
        print("Reject the null hypothesis: The means are significantly different")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different")







# ---------------- Similar to list, difficulty with parsing a list through functions, reproduced separately --------------------
# Thank you chatgpt for labor


# Distribution testing for Sepsis vs No sepsis
csv_file_path = 'avg_numeric_grouped_by_id+sepsislabel.csv'
shapiro_wilk_test_categorical(csv_file_path, 'avg_hr')
shapiro_wilk_test_categorical(csv_file_path, 'avg_o2sat')
shapiro_wilk_test_categorical(csv_file_path, 'avg_temp')
shapiro_wilk_test_categorical(csv_file_path, 'avg_sbp')
shapiro_wilk_test_categorical(csv_file_path, 'avg_map')
shapiro_wilk_test_categorical(csv_file_path, 'avg_dbp')
shapiro_wilk_test_categorical(csv_file_path, 'avg_resp')
shapiro_wilk_test_categorical(csv_file_path, 'avg_etco2')
shapiro_wilk_test_categorical(csv_file_path, 'avg_baseexcess')
shapiro_wilk_test_categorical(csv_file_path, 'avg_hco3')
shapiro_wilk_test_categorical(csv_file_path, 'avg_fio2')
shapiro_wilk_test_categorical(csv_file_path, 'avg_ph')
shapiro_wilk_test_categorical(csv_file_path, 'avg_paco2')
shapiro_wilk_test_categorical(csv_file_path, 'avg_sao2')
shapiro_wilk_test_categorical(csv_file_path, 'avg_ast')
shapiro_wilk_test_categorical(csv_file_path, 'avg_bun')
shapiro_wilk_test_categorical(csv_file_path, 'avg_alkalinephos')
shapiro_wilk_test_categorical(csv_file_path, 'avg_calcium')
shapiro_wilk_test_categorical(csv_file_path, 'avg_chloride')
shapiro_wilk_test_categorical(csv_file_path, 'avg_creatinine')
shapiro_wilk_test_categorical(csv_file_path, 'avg_bilirubin_direct')
shapiro_wilk_test_categorical(csv_file_path, 'avg_glucose')
shapiro_wilk_test_categorical(csv_file_path, 'avg_lactate')
shapiro_wilk_test_categorical(csv_file_path, 'avg_magnesium')
shapiro_wilk_test_categorical(csv_file_path, 'avg_phosphate')
shapiro_wilk_test_categorical(csv_file_path, 'avg_potassium')
shapiro_wilk_test_categorical(csv_file_path, 'avg_bilirubin_total')
shapiro_wilk_test_categorical(csv_file_path, 'avg_troponini')
shapiro_wilk_test_categorical(csv_file_path, 'avg_hct')
shapiro_wilk_test_categorical(csv_file_path, 'avg_hgb')
shapiro_wilk_test_categorical(csv_file_path, 'avg_ptt')
shapiro_wilk_test_categorical(csv_file_path, 'avg_wbc')
shapiro_wilk_test_categorical(csv_file_path, 'avg_fibrinogen')
shapiro_wilk_test_categorical(csv_file_path, 'avg_platelets')
shapiro_wilk_test_categorical(csv_file_path, 'avg_age')
shapiro_wilk_test_categorical(csv_file_path, 'avg_gender')
shapiro_wilk_test_categorical(csv_file_path, 'avg_iculos')


# Distribution testing for longitudinal data (vital-signs)
shapiro_wilk_test_temporal('temporal_vitals.csv', 'hr')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'o2')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'temp')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'sbp')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'map')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'dbp')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'resp')
shapiro_wilk_test_temporal('temporal_vitals.csv', 'etco2') # hour 0 = normal


# Distribution testing for longitudinal data (laboratory and categorical values)
csv_file_path = 'temporal_non_vitals.csv'
shapiro_wilk_test_temporal(csv_file_path, 'baseexcess') # both normal
shapiro_wilk_test_temporal(csv_file_path, 'hco3') # six hour <
shapiro_wilk_test_temporal(csv_file_path, 'fio2') #
shapiro_wilk_test_temporal(csv_file_path, 'ph') # both normal
shapiro_wilk_test_temporal(csv_file_path, 'paco2') # hour 0
shapiro_wilk_test_temporal(csv_file_path, 'sao2')
shapiro_wilk_test_temporal(csv_file_path, 'ast')
shapiro_wilk_test_temporal(csv_file_path, 'bun')
shapiro_wilk_test_temporal(csv_file_path, 'alkalinephos')
shapiro_wilk_test_temporal(csv_file_path, 'calcium')
shapiro_wilk_test_temporal(csv_file_path, 'chloride') # hour 0
shapiro_wilk_test_temporal(csv_file_path, 'creatinine')
#shapiro_wilk_test_temporal(csv_file_path, 'bilirubin_direct')
shapiro_wilk_test_temporal(csv_file_path, 'glucose')
shapiro_wilk_test_temporal(csv_file_path, 'lactate')
shapiro_wilk_test_temporal(csv_file_path, 'magnesium') # six hour
shapiro_wilk_test_temporal(csv_file_path, 'phosphate') # hour 0
shapiro_wilk_test_temporal(csv_file_path, 'potassium') # hour 0
shapiro_wilk_test_temporal(csv_file_path, 'bilirubin_total')
#shapiro_wilk_test_temporal(csv_file_path, 'troponini')
shapiro_wilk_test_temporal(csv_file_path, 'hct') # both
shapiro_wilk_test_temporal(csv_file_path, 'hgb') # both
shapiro_wilk_test_temporal(csv_file_path, 'ptt')
shapiro_wilk_test_temporal(csv_file_path, 'wbc') # hour 0
shapiro_wilk_test_temporal(csv_file_path, 'fibrinogen') # both
shapiro_wilk_test_temporal(csv_file_path, 'platelets') # hour 0





# No normal distributions found in Sepsis vs no sepsis group. Comparative analysis using significance testing.
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_hr')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_o2sat')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_temp')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_sbp')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_map')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_dbp')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_resp')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_etco2')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_baseexcess')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_hco3')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_fio2')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_ph')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_paco2')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_sao2')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_ast')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_bun')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_alkalinephos')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_calcium')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_chloride')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_creatinine')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_bilirubin_direct')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_glucose')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_lactate')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_magnesium')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_phosphate')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_potassium')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_bilirubin_total')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_troponini')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_hct')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_hgb')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_ptt')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_wbc')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_fibrinogen')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_platelets')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_age')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_gender')
mann_whitney_u_test('avg_numeric_grouped_by_id+sepsislabel.csv','avg_iculos')


wilcoxon_temporal('temporal_vitals.csv','hr')
wilcoxon_temporal('temporal_vitals.csv','o2')
wilcoxon_temporal('temporal_vitals.csv','temp')
wilcoxon_temporal('temporal_vitals.csv','sbp')
wilcoxon_temporal('temporal_vitals.csv','map')
wilcoxon_temporal('temporal_vitals.csv','dbp')
wilcoxon_temporal('temporal_vitals.csv','resp')


wilcoxon_temporal('temporal_non_vitals.csv','baseexcess')
wilcoxon_temporal('temporal_non_vitals.csv','hco3')
wilcoxon_temporal('temporal_non_vitals.csv','fio2')
wilcoxon_temporal('temporal_non_vitals.csv','ph')
wilcoxon_temporal('temporal_non_vitals.csv','paco2')
wilcoxon_temporal('temporal_non_vitals.csv','sao2')
wilcoxon_temporal('temporal_non_vitals.csv','ast')
wilcoxon_temporal('temporal_non_vitals.csv','bun')
wilcoxon_temporal('temporal_non_vitals.csv','alkalinephos')
wilcoxon_temporal('temporal_non_vitals.csv','calcium')
wilcoxon_temporal('temporal_non_vitals.csv','chloride')
wilcoxon_temporal('temporal_non_vitals.csv','creatinine')
wilcoxon_temporal('temporal_non_vitals.csv','bilirubin_direct')
wilcoxon_temporal('temporal_non_vitals.csv','glucose')
wilcoxon_temporal('temporal_non_vitals.csv','lactate')
wilcoxon_temporal('temporal_non_vitals.csv','magnesium')
wilcoxon_temporal('temporal_non_vitals.csv','phosphate')
wilcoxon_temporal('temporal_non_vitals.csv','potassium')
wilcoxon_temporal('temporal_non_vitals.csv','bilirubin_total')
wilcoxon_temporal('temporal_non_vitals.csv','troponini')
wilcoxon_temporal('temporal_non_vitals.csv','hct')
wilcoxon_temporal('temporal_non_vitals.csv','hgb')
wilcoxon_temporal('temporal_non_vitals.csv','ptt')
wilcoxon_temporal('temporal_non_vitals.csv','wbc')
wilcoxon_temporal('temporal_non_vitals.csv','fibrinogen')
wilcoxon_temporal('temporal_non_vitals.csv','platelets')

# Unequal variance (one normal - one not normal)
welchs_t_test('temporal_vitals.csv', 'etco2')
welchs_t_test('temporal_non_vitals.csv', 'hco3')
welchs_t_test('temporal_non_vitals.csv', 'paco2')
welchs_t_test('temporal_non_vitals.csv', 'chloride')
welchs_t_test('temporal_non_vitals.csv', 'magnesium')
welchs_t_test('temporal_non_vitals.csv', 'phosphate')
welchs_t_test('temporal_non_vitals.csv', 'potassium')
welchs_t_test('temporal_non_vitals.csv', 'wbc')
welchs_t_test('temporal_non_vitals.csv', 'platelets')


# Comparison of two normally distributed datasets with a longitudinal relationship
paired_samples_ttest('temporal_non_vitals.csv', 'baseexcess')
paired_samples_ttest('temporal_non_vitals.csv', 'ph')
paired_samples_ttest('temporal_non_vitals.csv', 'hgb')
paired_samples_ttest('temporal_non_vitals.csv', 'hct')
paired_samples_ttest('temporal_non_vitals.csv', 'fibrinogen')