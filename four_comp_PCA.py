
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# 4 components
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


def perform_pca(data, n_components=4):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)

    column_names = [f'Principal Component {i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result, columns=column_names)
    explained_variance = pca.explained_variance_ratio_

    loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'Principal Component {i+1}' for i in range(4)], 
                        index=durham_data.columns)
    
    
    return pca_df, explained_variance, loadings

durham_pca_df, durham_explained_variance,loadings = perform_pca(durham_data, n_components=4)
print("Durham PCA Results:")
print(durham_pca_df.head())
durham_pca_df.head().to_csv('four_comp_coeffs_durham.csv', index=True)

print(f"Explained Variance for Durham Data: {durham_explained_variance}")
loadings.to_csv('four_comp_pca_loadings_durham.csv', index=True)




statewide_pca_df, statewide_explained_variance, loadings = perform_pca(statewide_data, n_components=4)
print("Statewide PCA Results:")
print(statewide_pca_df.head())
print(f"Explained Variance for Statewide Data: {statewide_explained_variance}")
loadings.to_csv('four_comp_pca_loadings_nc.csv', index=True)
print("4 Comp PCA Loadings:\n", loadings)


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=durham_pca_df, alpha=0.7)
plt.title('PCA of Durham Data')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=statewide_pca_df, alpha=0.7)
plt.title('PCA of Statewide Data')

plt.tight_layout()
plt.show()





'''Insight:

Loadings for Durham:
,Principal Component 1,Principal Component 2,Principal Component 3,Principal Component 4
subject_age,0.9957696812130693,0.06503868636564189,0.026026196312564445,0.05945368748068318
subject_race_cat_black,-0.029394411985912014,0.7591422806169631,-0.04827818798948874,-0.32203113389801236
subject_race_cat_hispanic,-0.04424829782653144,-0.10619465536422187,0.18394478344275536,0.7711352077798109
subject_race_cat_unknown,-0.0002714136422379748,-0.0068772771537047896,0.007367332052988996,0.018253388733600197
subject_race_cat_white,0.07447681438603325,-0.637641911241887,-0.1456743133153003,-0.49122708650267627
subject_sex_cat_male,-0.008598204104487415,-0.03951149597682519,0.9705054802383585,-0.23764356177588758
constant,-2.910517311332564e-17,-1.768284017172459e-16,1.4742394764906394e-17,1.8218274857692575e-16

Loadings for State:

,Principal Component 1,Principal Component 2,Principal Component 3,Principal Component 4
subject_age,0.9978454497016055,-0.05582000241401537,-0.00999017100183863,0.032989836727657235
subject_race_cat_black,-0.022258564994141458,-0.6715581256290072,-0.045537459848267625,-0.4864618574063801
subject_race_cat_hispanic,-0.02765324503420679,-0.04824715623941749,0.09181375393944381,0.770912934231025
subject_race_cat_unknown,-0.002526317729502771,-0.011169846741838556,0.008172166247124826,0.06993360114844666
subject_race_cat_white,0.05313514848381707,0.7370207930268545,-0.05705784139827436,-0.38678216000426263
subject_sex_cat_male,0.014648772904684212,0.015543754912979024,0.993012784034157,-0.11605443519993269
constant,2.0685657634775184e-16,-2.850965399990086e-17,-9.111082925005596e-18,6.241516027644571e-17

Similar loadings results for the first 3 PCs 
Newest insight is that the race category (hispanic) variable heavily influenced PCA 4 (0.77) 
showing latino race as a driving factor for PC4.


'''




