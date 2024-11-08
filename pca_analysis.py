import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_feather('nc_reduced_data.feather')
print(df['file_source'].unique())

# Define all possible categories upfront
all_race_categories = ['black', 'white', 'hispanic', 'asian/pacific islander', 'unknown']
all_sex_categories = ['female', 'male', 'unknown']
all_outcome_categories = ['warning', 'citation', 'arrest', 'no_action']

# Split data
durham_data = df[df['file_source'] == 'nc_durham']
statewide_data = df[df['file_source'] == 'nc_statewide']

def encode_onhot(data):
    data['subject_race_cat'] = pd.Categorical(data['subject_race_cat'], categories=all_race_categories)
    data['subject_sex_cat'] = pd.Categorical(data['subject_sex_cat'], categories=all_sex_categories)
    data['outcome_cat'] = pd.Categorical(data['outcome_cat'], categories=all_outcome_categories)

    data['subject_race_cat'] = data['subject_race_cat'].replace(['unknown', 'other'], 'unknown').fillna('unknown')
    data['subject_sex_cat'] = data['subject_sex_cat'].fillna('unknown')
    data['outcome_cat'] = data['outcome_cat'].fillna('no_action')

    data = data.dropna(subset=['subject_age'])

    numeric_features = ['subject_age']
    categorical_features = ['subject_race_cat', 'subject_sex_cat', 'file_source']

    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(data[numeric_features])
    X_numeric_df = pd.DataFrame(X_numeric_scaled, columns=numeric_features)

    onehot = OneHotEncoder(drop='first', sparse_output=True)
    X_categorical_encoded = onehot.fit_transform(data[categorical_features])
    categorical_col_names = onehot.get_feature_names_out(categorical_features)
    X_categorical_df = pd.DataFrame.sparse.from_spmatrix(X_categorical_encoded, columns=categorical_col_names)

    X_combined = pd.concat([X_numeric_df.reset_index(drop=True), X_categorical_df.reset_index(drop=True)], axis=1)
    X_combined['constant'] = 1

    return X_combined

durham_data = encode_onhot(durham_data)
statewide_data = encode_onhot(statewide_data)

print(len(durham_data))


def perform_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)

    column_names = [f'Principal Component {i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result, columns=column_names)
    explained_variance = pca.explained_variance_ratio_

    loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'Principal Component {i+1}' for i in range(3)], 
                        index=durham_data.columns)
    
    
    return pca_df, explained_variance, loadings

durham_pca_df, durham_explained_variance,loadings = perform_pca(durham_data, n_components=3)
print("Durham PCA Results:")
print(durham_pca_df.head())
print(f"Explained Variance for Durham Data: {durham_explained_variance}")
loadings.to_csv('pca_loadings_durham.csv', index=True)

statewide_pca_df, statewide_explained_variance, loadings = perform_pca(statewide_data, n_components=3)
print("Statewide PCA Results:")
print(statewide_pca_df.head())
print(f"Explained Variance for Statewide Data: {statewide_explained_variance}")
loadings.to_csv('pca_loadings_nc.csv', index=True)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=durham_pca_df, alpha=0.7)
plt.title('PCA of Durham Data')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=statewide_pca_df, alpha=0.7)
plt.title('PCA of Statewide Data')

plt.tight_layout()
plt.show()
'''
Durham Results:

   Principal Component 1  Principal Component 2  Principal Component 3
0              -1.025133              -0.501178               0.644931
1              -0.582993               0.373260              -0.569967
2              -1.111330              -0.456598              -0.323547
3              -0.559541              -0.531588               0.632761
4               0.061249              -0.572135               0.616536
Explained Variance for Durham Data: [0.55756544 0.218913   0.13065647]



,Principal Component 1,Principal Component 2,Principal Component 3
subject_age,0.9957696812130694,-0.06503868636563667,-0.0260261963125652
subject_race_cat_black,-0.029394411985908447,-0.7591422806169642,0.04827818798947714
subject_race_cat_hispanic,-0.04424829782653178,0.10619465536422412,-0.18394478344275494
subject_race_cat_unknown,-0.0002714136422380059,0.006877277153704896,-0.0073673320529888
subject_race_cat_white,0.07447681438603006,0.6376419112418845,0.14567431331531075
subject_sex_cat_male,-0.008598204104487334,0.039511495976840186,-0.9705054802383571
constant,0.0,0.0,0.0

State Results:


   Principal Component 1  Principal Component 2  Principal Component 3
0               0.766039              -0.478425              -0.328536
1              -1.070674              -0.581171              -0.346925
2              -0.931057               0.216426              -0.493590
3               1.868067              -0.416777              -0.317503
4              -0.556394              -0.552402              -0.341776
Explained Variance for Statewide Data: [0.56509753 0.22872361 0.12989527]

,Principal Component 1,Principal Component 2,Principal Component 3
subject_age,0.9978454497015976,0.05582000241415727,0.009990171001807364
subject_race_cat_black,-0.022258564994232483,0.671558125629051,0.04553745984767282
subject_race_cat_hispanic,-0.027653245034210912,0.04824715623932696,-0.09181375393951718
subject_race_cat_unknown,-0.00252631772950406,0.01116984674183056,-0.008172166247138306
subject_race_cat_white,0.05313514848391361,-0.73702079302679,0.05705784139896508
subject_sex_cat_male,0.014648772904707117,-0.015543754913883878,-0.993012784034138
constant,0.0,-0.0,-0.0


Insight:



From the loading results of both Durham and statewide (NC) data, it is evident that subject_age is the most significant contributor to Principal Component 1 (PC1), with coefficients close to 1 in both datasets (0.9958 for Durham and 0.9978 for NC). This shows that age is the primary factor influencing the variance captured by PC1. For Principal Component 2 (PC2), the Durham data shows significant contributions from subject_race_cat_black with a negative loading of -0.7591 and subject_race_cat_white with a positive loading of 0.6376, suggesting that PC2 reflects variability related to racial demographics, particularly between black and white individuals. Similarly, in the NC data, PC2 shows significant loadings from subject_race_cat_black (0.6716) and subject_race_cat_white (-0.7370), indicating that this component also captures contrasts between these racial categories but with opposite signs compared to Durham. This sign difference may suggest differing patterns of racial contributions to data variability at the state and local levels.

For Principal Component 3 (PC3), the most influential feature in both datasets is subject_sex_cat_male, with negative loadings of -0.9705 for Durham and -0.9930 for NC, indicating that PC3 is predominantly driven by gender, highlighting male as a significant factor in this component. The constant feature consistently shows a loading of 0.0 in all components for both datasets, confirming it does not contribute to the variance captured by the principal components. 

Overall, PC1 is primarily driven by age, making it a strong age-related component across both datasets, while PC2 emphasizes racial categories with differing local and statewide patterns, and PC3 predominantly captures gender-based variance. These results point to age as the main source of variance, followed by racial and gender influences, which may suggest socio-demographic and procedural differences that warrant further investigation for deeper insights.


'''