"""Example of how to call the different bricks on the PRA dataset."""

import numpy as np
import pandas as pd

from pre_linkage_metrics import ImputeMethod, contribution_score, get_best_n_from_m_variables, get_unicity_score
from linkage import Distance, link_datasets, LinkingAlgorithm


# load avatars from both sources
df1_avatars = pd.read_csv("data/afi_A_unshuffled_avatars.csv")
df2_avatars = pd.read_csv("data/afi_B_unshuffled_avatars.csv")

should_be_categorical_columns_1 = ['nationalite', 'pays_residence', 'sexe']
should_be_categorical_columns_2 = ['nationalite', 'pays_residence', 'sexe', 'demande_statut', 'diplome', 'domaine_formation', 'pays_etablissement']

for col in should_be_categorical_columns_1:
    df1_avatars[col] = df1_avatars[col].astype(object)
for col in should_be_categorical_columns_2:
    df2_avatars[col] = df2_avatars[col].astype(object)

# Select common/shared columns
shared_columns = should_be_categorical_columns_1

linkage_algos = [LinkingAlgorithm.LSA, LinkingAlgorithm.MIN_ORDER]
# linkage_algos = [LinkingAlgorithm.LSA]
distances = [Distance.GOWER, Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_SECOND_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.ROW_ORDER, Distance.RANDOM]
# distances = [Distance.PROJECTION_DIST_FIRST_SOURCE, Distance.PROJECTION_DIST_ALL_SOURCES, Distance.RANDOM]
# linkage_algos = [LinkingAlgorithm.MIN_ORDER]
# distances = [Distance.RANDOM]
should_shuffle_before_linkage = True

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
