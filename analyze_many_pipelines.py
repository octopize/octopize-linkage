import pandas as pd

file_path = "data/many_pipeline_stats_20241120_103839.csv"

post_linkage_metric = "correlation_difference_mean"
# post_linkage_metric = "reconstruction_difference_mean"

pre_linkage_metrics = "unicity_score1"
# pre_linkage_metrics = "contribution_score1"



stats_df = pd.read_csv(file_path)
# stats_df['correlation_retention'] = 100 - stats_df['corr_diff_sum']

# reference_corr_diff_sums = {combination_id: min(stats_df[(stats_df['combination_id'] == combination_id) & (stats_df['distance'] == 'row_order')]['corr_diff_sum']) for combination_id in stats_df['combination_id'].unique()}
# print(reference_corr_diff_sums)

# # create new column with reference_corr_diff_sums for each combination_id
# stats_df['reference_corr_diff_sum'] = stats_df['combination_id'].map(reference_corr_diff_sums)

# # create new column with relative percentage difference between corr_diff_sum and reference_corr_diff_sum
# stats_df['correlation_retention'] = - abs((stats_df['corr_diff_sum'] - stats_df['reference_corr_diff_sum']))

print(stats_df)

# TODO: make the correlation score as a percentage between 0 (random) and 100 (row order)






import seaborn as sns
import matplotlib.pyplot as plt


##########################################
### Questions asked?
### - how close to an ideal linkage (row order) can we get with the proposed methods?
###     - and how far are we from random linkage? 
### - what is the correlation between post and pre linkage metrics?
###     - using unicity_score1 and correlation_retention
### - what is the impact of avatarization and linkage (comparing linkage done on avatars and original data)
###    - done by comparing the plot obtained by linking avatars and original
##########################################

data = stats_df[stats_df['ava_ori'] == 'avatars']
sns.lmplot(data=data, x=pre_linkage_metrics, y=post_linkage_metric, hue='distance', scatter=True)
plt.xlabel(pre_linkage_metrics)
plt.ylabel(post_linkage_metric)
plt.title(f'{post_linkage_metric} vs {pre_linkage_metrics} for Each Distance on Avatars')
plt.legend(title='Linkage Algorithm')
plt.show()


data = stats_df[stats_df['ava_ori'] == 'original']
sns.lmplot(data=data, x=pre_linkage_metrics, y=post_linkage_metric, hue='distance', scatter=True)
plt.xlabel(pre_linkage_metrics)
plt.ylabel(post_linkage_metric)
plt.title(f'{post_linkage_metric} vs {pre_linkage_metrics} for Each Distance on Originals')
plt.legend(title='Linkage Algorithm')
plt.show()



##########################################
### Questions asked?
### - how close to an ideal linkage (row order) can we get with the proposed methods?
###     - and how far are we from random linkage? 
### - what is the correlation between post and pre linkage metrics?
###     - using contribution_score1 and correlation_retention
### - what is the impact of avatarization and linkage (comparing linkage done on avatars and original data)
###    - done by comparing the plot obtained by linking avatars and original
##########################################

# data = stats_df[stats_df['ava_ori'] == 'avatars']
# sns.lmplot(data=data, x='contribution_score1', y='correlation_retention', hue='distance', scatter=True)

# plt.xlabel('Contribution Score')
# plt.ylabel('Correlation Retention')
# plt.title('Correlation_retention vs Contribution Score for Each Distance on Avatars')
# plt.legend(title='Linkage Algorithm')
# plt.show()


# data = stats_df[stats_df['ava_ori'] == 'original']
# sns.lmplot(data=data, x='contribution_score1', y='correlation_retention', hue='distance', scatter=True)

# plt.xlabel('Contribution Score')
# plt.ylabel('Correlation Retention')
# plt.title('Correlation_retention vs Contribution Score for Each Distance on Originals')
# plt.legend(title='Linkage Algorithm')
# plt.show()



##########################################
### Questions asked?
### - is there a correlation between the the two pre-linkage scores?
##########################################

data = stats_df
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='unicity_score1', y='contribution_score1', hue='distance')
plt.xlabel('Unicity score')
plt.ylabel('Contribution score')
plt.title('Unicity score vs. Contribution score')
plt.show()


##########################################
### What is the impact of avatarization and linkage (comparing linkage done on avatars and original data)
##########################################



# generate a plot with x being the unicity_score1 and y being the correlation_retention and tracing a line between original and avatar for each group made of {combination_id, distance, linkage_algo}
plt.figure(figsize=(10, 6))

# Group by combination_id, distance, and linkage_algo
data = stats_df[(stats_df['distance'] == 'proj_eucl_all_source') & (stats_df['linkage_algo'] == 'lsa')]
grouped = data.groupby(['combination_id', 'distance', 'linkage_algo'])
# grouped = stats_df.groupby('combined_name')

markers_dict = {
    'original': 'o',
    'avatars': 's',
}
# Plot each group
for name, group in grouped:
    # sns.lineplot(data=group, x='unicity_score1', y='correlation_retention', style='ava_ori', markers=markers_dict, label=name)
    # sns.lineplot(data=group, x='unicity_score1', y='correlation_retention', style='ava_ori', markers=markers_dict)
    sns.lineplot(data=group, x=pre_linkage_metrics, y=post_linkage_metric, markers=markers_dict, style='ava_ori')
    # sns.lineplot(data=group, x='unicity_score1', y='correlation_retention', marker='o', label=name)
    # Add arrowhead
    plt.annotate('', xy=(group[pre_linkage_metrics].iloc[-1], group[post_linkage_metric].iloc[-1]), 
                 xytext=(group[pre_linkage_metrics].iloc[-2], group[post_linkage_metric].iloc[-2]),
                 arrowprops=dict(arrowstyle="->", color='blue'))

plt.xlabel(pre_linkage_metrics)
plt.ylabel(post_linkage_metric)
plt.title(f'{post_linkage_metric} for different data splits\n showing linkage done on avatars and original data\n (original --> avatar)')
plt.legend([],[], frameon=False)  # Hide the legend

plt.show()




##########################################
# Plot a 3D plot with x being the unicity_score1, y being the contribution_score1, and z being the correlation_retention
##########################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming stats_df is already defined and loaded

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')




# Plot the data points
# data = stats_df[(stats_df['distance'] == 'proj_eucl_all_source') & (stats_df['linkage_algo'] == 'lsa') & (stats_df['ava_ori'] == 'avatars')]
# data = stats_df[(stats_df['distance'] == 'gower') & (stats_df['linkage_algo'] == 'lsa') & (stats_df['ava_ori'] == 'avatars')]
data = stats_df[(stats_df['linkage_algo'] == 'lsa') & (stats_df['ava_ori'] == 'avatars')]

# Create a color palette
palette = sns.color_palette("hsv", len(data['distance'].unique()))
color_map = dict(zip(data['distance'].unique(), palette))

# Plot the data points with different colors based on the distance
for distance in data['distance'].unique():
    subset = data[data['distance'] == distance]
    ax.scatter(subset['unicity_score1'], subset['contribution_score1'], subset[post_linkage_metric], 
               c=[color_map[distance]], label=distance, marker='o')


# ax.scatter(data['unicity_score1'], data['contribution_score1'], data['correlation_retention'], c='b', marker='o')

# Set labels
ax.set_xlabel(pre_linkage_metrics)
ax.set_ylabel('Contribution Score')
ax.set_zlabel(post_linkage_metric)
ax.set_title(f'3D Plot of {pre_linkage_metrics}, Contribution Score, and {post_linkage_metric}')

# Show legend
ax.legend(title='Distance')

plt.show()