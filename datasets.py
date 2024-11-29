import os
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def download_task(dataset_name, preprocess=True):
    import lale
    cached_data_path = f"data/{dataset_name}_{preprocess}.pkl"
    print(cached_data_path)
    if not os.path.exists(cached_data_path):
        load_df = getattr(lale.lib.aif360.datasets, 'fetch_'+dataset_name+'_df')
        X, y, fairness_info =  load_df()
        print(fairness_info)
        l = fairness_info['protected_attributes']
        sens_names = [d['feature'] for d in l]

        print("Downloaded")
        X.reset_index(drop=True)
        y.reset_index(drop=True)
        
        print(X.shape)
        print(y.shape)
        print(X.head(10))

        if preprocess:
            # Minimal preprocessing

            if dataset_name=='compas_violent':
                # Identify the date columns
                date_columns = ['compas_screening_date', 'dob', 'in_custody', 'out_custody', 'v_screening_date', 'c_jail_in', 'c_jail_out', 'c_offense_date', 'c_arrest_date', 'vr_offense_date', 'screening_date',]
                for col in date_columns:
                    # Drop the date columns
                    X = X.drop(col, axis=1)

            if dataset_name=='default_credit':
                X['sex'] = w=X['sex'].map({1:'one', 2:'two'})
                
            for col in X:
                if len(X[col].unique())==1:
                    X = X.drop(col, axis=1)
        
            # If any sensitive feature column contains continuous values (e.g. age), bin it.
            if 'age' in sens_names:
                bin_edges = [0, 18, 35, 50, float('inf')]  # Define your desired age bins
                bin_labels = ['0-18', '19-35', '36-50', '51+']  # Labels for the bins
            
                # Create a new column 'age_group' based on the bins
                X['age'] = pd.cut(X['age'], bins=bin_edges, labels=bin_labels, right=False)
               
            fav_label = fairness_info["favorable_labels"][0]
            print("Before")
            print(y.head(20))
            y = pd.Series([1 if y==fav_label else 0 for y in y])
            print("After")
            print(y.head(20))
           
            features = X.columns

            print("All features", features)
            print("Sensitive features", sens_names)

            assert y.index.equals(X.index), "Indices of y and sensitive_columns do not match."
            d = {"X": X, "y": y, "features":features, "sens_features":sens_names}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)


def download_pmad_task(dataset_name, outcome_name, preprocess=True):
    cached_data_path = f"Datasets/{dataset_name}_{preprocess}.pkl"
    print(cached_data_path)
    if not os.path.exists(cached_data_path):
        all_data = pd.read_excel("De-identified PMAD data.xlsx")

        # Extract relevant variables for model fitting
        outcome = outcome_name
        data = all_data[['MOM_AGE','MOM_RACE','ETHNIC_GROUP','MARITAL_STATUS','FINANCIAL_CLASS',
                        'LBW','PTB',
                        'DELIVERY_METHOD','NICU_ADMIT','MFCU_ADMIT',
                        'PREE','GDM','GHTN',
                        'MOM_BMI','MOM_LOS','CHILD_LOS',
                        'HIST_ANXIETY','HIST_DEPRESS','HIST_BIPOLAR','HIST_PMAD','MENTAL_HEALTH_DX_CUTOFF',
                        'MED_PSYCH','MED_CARDIO',
                        outcome]]

        data = data.dropna() # keep only complete data

        # get dummy variables
        data = pd.get_dummies(data, dtype=int)

        # split into X and y
        X = data.drop([outcome], axis=1)
        y = data[[outcome]]

        features = X.columns

        sens_features = ['MOM_RACE']
        print("All features", features)
        print("Sensitive features", sens_features)
        y = y.squeeze().astype('int')
        assert y.index.equals(X.index), "Indices of y and sensitive_columns do not match."
        d = {"X": X, "y": y, "features":features, "sens_features":sens_features}
        if not os.path.exists("Datasets"):
            os.makedirs("Datasets")
        with open(cached_data_path, "wb") as f:
            pickle.dump(d, f)



if __name__ == '__main__':
    #not_able_to_download = ['meps19', 'meps21', 'meps20',]
    datasets_binary = ['heart_disease', 'student_math', 'us_crime', 'nlsy', 'compas', 'law_school']
    for ds in datasets_binary:
        download_task(ds)
    # The following two medical datasets require special permission from Cedars-Sinai Medical Center. Contact the authors for access.
    #download_pmad_task('pmad_phq','PHQ9_risk2')
    #download_pmad_task('pmad_epds','EPDS_risk2')


