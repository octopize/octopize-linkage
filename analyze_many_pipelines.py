import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# file_path = "data/many_pipeline_stats_20241120_170813.csv"  # full results
file_path = "data/many_sizes_stats_20241121_144048.csv"  # many sizes only 

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

# concat the two columns linkage_algo and distance
stats_df['method'] = stats_df['linkage_algo'] + '_' + stats_df['distance']

stats_df['mean_unicity_score'] = (stats_df['unicity_score1'] + stats_df['unicity_score2']) / 2
stats_df['mean_contribution_score'] = (stats_df['contribution_score1'] + stats_df['contribution_score2']) / 2

print(stats_df)

# TODO: make the correlation score as a percentage between 0 (random) and 100 (row order)

MINIMUM_MEAN_UNICITY_SCORE = 0.5

datasets = stats_df['dataset'].unique()



# Create a seaborn boxplot with mean unicity score per dataset
sns.boxplot(data=stats_df, x='dataset', y='mean_unicity_score')

# Set labels and title
plt.xlabel('Dataset')
plt.ylabel('Mean Unicity Score')
plt.title('Mean Unicity Score for Each Dataset')

# Show the plot
plt.tight_layout()
plt.show()




# Define the datasets
datasets = stats_df['dataset'].unique()

# Create a figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Define a color palette
palette = sns.color_palette('tab10', n_colors=len(stats_df['number_records'].unique()))
color_map = {record: palette[i] for i, record in enumerate(stats_df['number_records'].unique())}

for i, dataset in enumerate(datasets):
    ax = axes[i]
    data = stats_df[stats_df['dataset'] == dataset]
    
    # Create the scatter plot with different colors for each number_records
    sns.scatterplot(data=data, x='mean_unicity_score', y=post_linkage_metric, hue='number_records', palette=color_map, ax=ax)
    
    # Add regression lines for each number_records
    unique_records = data['number_records'].unique()
    for record in unique_records:
        subset = data[data['number_records'] == record]
        sns.regplot(data=subset, x='mean_unicity_score', y=post_linkage_metric, scatter=False, ax=ax, color=color_map[record])
    
    # Set labels and title
    ax.set_xlabel('Mean Unicity Score')
    ax.set_ylabel(post_linkage_metric)
    ax.set_title(f'{dataset}')
    ax.legend(title='Dataset Size (number of records)')

# Set the overall title
plt.suptitle(f'Correlation between Mean Unicity Score and {post_linkage_metric} for linkage of avatars and different dataset sizes')

# Adjust layout
plt.tight_layout()
plt.show()




# Create a suplot for each dataset with the correlation between mean contribution score and post_linkage_metric
ncols = 2 
nrows = math.ceil(len(datasets)/2)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, dataset in enumerate(datasets):
    row = i // ncols
    col = i % ncols
    data = stats_df[(stats_df['dataset'] == dataset)]
    sns.scatterplot(data=data, x='mean_unicity_score', y=post_linkage_metric, ax=axes[i], hue='number_records', palette='tab10')
    axes[i].set_title(f'{dataset}')
    axes[i].set_xlabel('Mean Unicity Score')
    axes[i].set_ylabel(post_linkage_metric)
    axes[i].legend(title='Dataset Size (number of records)')
plt.suptitle(f'Correlation between Mean unicity Score and {post_linkage_metric} for linkage of avatars and different dataset sizes')

plt.tight_layout()
plt.show()






nrows=2
ncols=1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, ori_ava in enumerate(stats_df['ava_ori'].unique()):
    data = stats_df[(stats_df['ava_ori'] == ori_ava) & (stats_df['mean_unicity_score'] > MINIMUM_MEAN_UNICITY_SCORE)]
    sns.boxplot(data=data, x='dataset', y=post_linkage_metric, hue='method', ax=axes[i])
    axes[i].set_title(f'{post_linkage_metric} for {ori_ava}')
    axes[i].set_xlabel('Dataset')
    axes[i].set_ylabel(post_linkage_metric)
    axes[i].legend(title='Linkage Algorithm')

plt.tight_layout()
plt.show()



# Create a suplot for each dataset with the correlation between mean unicity score and post_linkage_metric
ncols = 2 
nrows = math.ceil(len(datasets)/2)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, dataset in enumerate(datasets):
    row = i // ncols
    col = i % ncols
    data = stats_df[(stats_df['dataset'] == dataset) & (stats_df['method'] == 'lsa_proj_eucl_all_source')]
    sns.scatterplot(data=data, x='mean_unicity_score', y=post_linkage_metric, ax=axes[row, col], hue='ava_ori')
    axes[row, col].set_title(f'{dataset}')
    axes[row, col].set_xlabel('Mean Unicity Score')
    axes[row, col].set_ylabel(post_linkage_metric)
    # axes[row, col].legend(title='Linkage Algorithm')
# add title
plt.suptitle(f'Correlation between Mean Unicity Score and {post_linkage_metric} for best linkage method')
plt.tight_layout()
plt.show()




# Create a suplot for each dataset with the correlation between mean contribution score and post_linkage_metric
ncols = 2 
nrows = math.ceil(len(datasets)/2)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, dataset in enumerate(datasets):
    row = i // ncols
    col = i % ncols
    data = stats_df[(stats_df['dataset'] == dataset) & (stats_df['method'] == 'lsa_proj_eucl_all_source')]
    sns.scatterplot(data=data, x='mean_contribution_score', y=post_linkage_metric, ax=axes[row, col], hue='ava_ori')
    axes[row, col].set_title(f'{dataset}')
    axes[row, col].set_xlabel('Mean Contribution Score')
    axes[row, col].set_ylabel(post_linkage_metric)
    # axes[row, col].legend(title='Linkage Algorithm')
plt.suptitle(f'Correlation between Mean Contribution Score and {post_linkage_metric} for best linkage method')

plt.tight_layout()
plt.show()







# Create a suplot for each dataset with the correlation between mean contribution score and post_linkage_metric
ncols = 2 
nrows = math.ceil(len(datasets)/2)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, dataset in enumerate(datasets):
    row = i // ncols
    col = i % ncols
    data = stats_df[(stats_df['dataset'] == dataset)]
    sns.scatterplot(data=data, x='mean_unicity_score', y=post_linkage_metric, ax=axes[row, col], hue='number_records')
    axes[row, col].set_title(f'{dataset}')
    axes[row, col].set_xlabel('Mean Unicity Score')
    axes[row, col].set_ylabel(post_linkage_metric)
    axes[row, col].legend(title='Dataset Size (number of records)')
plt.suptitle(f'Correlation between Mean unicity Score and {post_linkage_metric} for linkage of avatars and different dataset sizes')

plt.tight_layout()
plt.show()

# # Create the seaborn boxplot
# sns.boxplot(data=stats_df, x='dataset', y=post_linkage_metric, hue='method')

# # Set labels and title
# plt.xlabel('Distance')
# plt.ylabel(post_linkage_metric)
# plt.title(f'{post_linkage_metric} for Each Distance and Dataset')
# plt.legend(title='Dataset')

# # Show the plot
# plt.tight_layout()
# plt.show()

assert False



# create 1 subplot per dataset, using 2 columns
ncols = 2
nrows = math.ceil(len(datasets)/2)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, dataset in enumerate(datasets):
    row = i // ncols
    col = i % ncols
    data = stats_df[stats_df['dataset'] == dataset]
    sns.boxplot(data=data, x='distance', y=post_linkage_metric, hue='method', ax=axes[row, col])
    axes[row, col].set_title(f'{post_linkage_metric} for {dataset}')
    axes[row, col].set_xlabel('Distance')
    axes[row, col].set_ylabel(post_linkage_metric)
    axes[row, col].legend(title='Linkage Algorithm')

plt.tight_layout()
plt.show()

assert False


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