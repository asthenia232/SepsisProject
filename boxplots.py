import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_visualisation_temporal(csv_file,  column, column1, show_outliers=True):
    df1 = pd.read_csv('avg_numeric_grouped_by_id+sepsislabel.csv')
    df = pd.read_csv(csv_file)


    sepsis_0_data_1 = df1[df1['sepsislabel'] == 0]
    six_hr_pre_data = df[f'{column}_six_hour_before_admission']
    post_admission_data = df[f'{column}_postadmission_hour0_sepsis']

    plt.figure(figsize=(10, 8))  # Set the figure size before creating the box plot
    ax = sns.boxplot(data=[sepsis_0_data_1[column1], six_hr_pre_data, post_admission_data ], palette=[ 'red', 'blue', 'orange'], showfliers=show_outliers)

    # Label and legend adjustments (legend not perfect)
    legend = ax.legend(labels=['no sepsis group', '6 Hours prior', 'Post-admission'])
    legend.set_bbox_to_anchor((1, 1))  # Adjust the anchor point as needed
    legend.get_frame().set_linewidth(2)  # Set the border width of the legend box

    ax.set_xlabel('Time')
    ax.set_ylabel(f'{column}')
    plt.title(f'Box Plot for {column}')
    plt.show()






# Same as above but modified for sepsis vs no sepsis
def boxplot_visualisation_one_feature(column):
    df = pd.read_csv('avg_numeric_grouped_by_id+sepsislabel.csv')

    # Filter data
    sepsis_0_data = df[df['sepsislabel'] == 0]
    sepsis_1_data = df[df['sepsislabel'] == 1]

    # Create a single plot with both box plots
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=[sepsis_0_data[column], sepsis_1_data[column]], palette=['blue', 'orange'])

    # Set labels for the legend
    legend = ax.legend(labels=['Non-Sepsis', 'Sepsis'])
    legend.set_bbox_to_anchor((1, 1))  # Adjust the anchor point as needed
    legend.get_frame().set_linewidth(2)  # Set the border width of the legend box

    ax.set_xlabel('Sepsis Label')
    ax.set_ylabel(f'{column}')
    plt.title(f'Box Plot for {column}')
    plt.show()




boxplot_visualisation_temporal('temporal_vitals.csv','hr', 'avg_hr', show_outliers=False)
boxplot_visualisation_temporal('temporal_vitals.csv','o2', 'avg_o2sat', show_outliers=False)
boxplot_visualisation_temporal('temporal_vitals.csv','temp', 'avg_temp', show_outliers=False)
boxplot_visualisation_temporal('temporal_vitals.csv','sbp', 'avg_sbp', show_outliers=False)
boxplot_visualisation_temporal('temporal_vitals.csv','map', 'avg_map', show_outliers=False)
boxplot_visualisation_temporal('temporal_non_vitals.csv','baseexcess', 'avg_baseexcess', show_outliers=False)
boxplot_visualisation_temporal('temporal_non_vitals.csv','fio2', 'avg_fio2', show_outliers=False)
boxplot_visualisation_temporal('temporal_non_vitals.csv','ph', 'avg_ph', show_outliers=False)
boxplot_visualisation_temporal('temporal_non_vitals.csv','sao2', 'avg_sao2', show_outliers=False)
boxplot_visualisation_temporal('temporal_non_vitals.csv','hco3', 'avg_hco3', show_outliers=False)







"""
-	Sepsis vs no sepsis – mann u whitney t-test
o	hr
o	temp
o	sbp
o	map
o	dbp
o	resp
o	etco2
o	base-excess
o	hco3
o	ph
o	ast
o	bun
o	akalinephos
o	calcium
o	creatinine
o	bilirubin
o	glucose
o	magnesium
o	phosphate
o	bilirubin_total
o	tropnini
o	hct
o	hgb
o	ptt
o	wbc
o	fibrinogen
o	platelets
o	gender
o	iculos



-	post administration vs 6 hours prior
o	hr
o	o2
o	temp
o	sbp
o	map
o	base excess – paired t-test
o	fio2
o	ph - paired t-test
o	sao2
o	hco3 – welch t-test
"""
