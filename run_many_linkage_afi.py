"""Script performing linkage under different settings for a given dataset."""

import pandas as pd

from linkage import Distance, link_datasets, LinkingAlgorithm
from pre_linkage_metrics import ImputeMethod, contribution_score, get_unicity_score
from post_linkage_metrics import generate_projection_plot, get_correlation_retention, get_correlations, get_non_shared_var_projections, get_reconstruction_score, plot_correlations

# load original data
df = pd.read_csv("data/afi_clear.csv")
df1 = pd.read_csv("data/afi_A.csv")
df2 = pd.read_csv("data/afi_B.csv")

# load avatars from both sources
df1_avatars = pd.read_csv("data/afi_A_unshuffled_avatars.csv")
df2_avatars = pd.read_csv("data/afi_B_unshuffled_avatars.csv")

should_be_categorical_columns_1 = ['nationalite', 'pays_residence', 'sexe']
should_be_categorical_columns_2 = ['nationalite', 'pays_residence', 'sexe', 'demande_statut', 'diplome', 'domaine_formation', 'pays_etablissement']

for col in should_be_categorical_columns_1:
    df1[col] = df1[col].astype(object)
    df1_avatars[col] = df1_avatars[col].astype(object)
for col in should_be_categorical_columns_2:
    df2[col] = df2[col].astype(object)
    df2_avatars[col] = df2_avatars[col].astype(object)

# Select common/shared columns
shared_columns = should_be_categorical_columns_1

#linkage_algos = [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER]
linkage_algos = [LinkingAlgorithm.LSA]
linkage_algo = "lsa"

#distances = [Distance.PROJECTION_DIST_ALL_SOURCES]
#distance = "proj_all_sources"
distances = [Distance.GOWER]
distance = "gower"
# distances = [Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.RANDOM]
# linkage_algos = [LinkingAlgorithm.MIN_ORDER]
# distances = [Distance.RANDOM]
should_shuffle_before_linkage = True

n_unique_comb1 = get_unicity_score(df1, shared_columns)
n_unique_comb2 = get_unicity_score(df2, shared_columns)

n_unique_comb1_avatar = get_unicity_score(df1_avatars, shared_columns)
n_unique_comb2_avatar = get_unicity_score(df2_avatars, shared_columns)

print("== Pre-linkage metrics ==")
print(f"Unicity score for source 1 original: {n_unique_comb1}")
print(f"Unicity score for source 2 original: {n_unique_comb2}")

print(f"Unicity score for source 1 avatar: {n_unique_comb1_avatar}")
print(f"Unicity score for source 2 avatar: {n_unique_comb2_avatar}")


for linkage_algo in linkage_algos:
    for distance in distances:
        _df1_avatars = df1_avatars.copy()
        _df2_avatars = df2_avatars.copy()
        if distance != Distance.ROW_ORDER and should_shuffle_before_linkage:
            _df2_avatars = _df2_avatars.sample(frac=1).reset_index(drop=True)
        print(f"\n\nRunning with linkage algorithm: {linkage_algo.value}, distance: {distance.value} ...")
        # link the two sources
        linked_df = link_datasets(_df1_avatars, _df2_avatars, shared_columns, distance=distance, linking_algo=linkage_algo)
        linked_df.to_csv(f"data/afi_linked_data__avatar__{linkage_algo.value}__{distance.value}.csv", index=False)
        # print("\n\nLinked data: \n", linked_df)


################################
# Post linkage Metrics
################################
_columns1 = set(df1_avatars.columns) - set(shared_columns)
_columns2 = set(df2_avatars.columns) - set(shared_columns)

linked_df = pd.read_csv("data/afi_linked_data__avatar__lsa__gower.csv")

# statistics
corr_records, corr_avatars = get_correlations(df, linked_df, list(_columns1), list(_columns2))

corr_retention_stats = get_correlation_retention(df, linked_df, list(_columns1), list(_columns2))
# {'corr_diff_sum': 17.497529801608245, 'corr_diff_mean': 0.019638080585418907, 'corr_diff_std': 0.0401587750859927, 'corr_diff_max': 0.8891200328461123}

reconstruction_stats = get_reconstruction_score(df, linked_df)
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 45.1 GiB for an array with shape (495962, 97619) and data type bool

# plots
plt = plot_correlations(corr_records, corr_avatars, title=f"{linkage_algo} / {distance} / afi")
plt.savefig(f"data/afi_linked_data__avatar__{linkage_algo}__{distance}_correlations.png")

proj_original, proj_linked = get_non_shared_var_projections(df, linked_df, list(_columns1), list(_columns2))
plt = generate_projection_plot(proj_original, proj_linked, title=f"Projection of non-shared variables for {linkage_algo}/{distance}")
plt.savefig(f"data/afi_linked_data__avatar__{linkage_algo}__{distance}_non_shared_var_projections.png")
