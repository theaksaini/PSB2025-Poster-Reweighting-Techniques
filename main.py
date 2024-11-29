from sklearn.ensemble import RandomForestClassifier

import experimental_setup


def main():
    ml_models = [RandomForestClassifier]
    # Note: The last two datasets in the following list are medical datasets and they require special permission from Cedars-Sinai Medical Center. 
    # Contact the authors for access.
    datasets_binary = ['heart_disease', 'student_math', 'us_crime', 'nlsy', 'compas', 'law_school','pmad_phq', 'pmad_epds']    
    experiments1 = ['Equal Weights', 'Deterministic Weights', 'Evolved Weights']

    gp_params_remote = {'pop_size':100, 'max_gens':50,  'mut_rate':0.1, 'cross_rate':0.8}

    data_dir = '../Datasets'
    # Choose one of the following two:
    #objective_functions=['accuracy', 'subgroup_FNR_loss']
    objective_functions=['accuracy', 'demographic_parity_difference']
     

    experimental_setup.loop_with_equal_evals2(ml_models= ml_models[0:1],
                                experiments=experiments1,
                                task_id_lists=datasets_binary,
                                base_save_folder='results',
                                   data_dir = data_dir,
                                num_runs=20,
                                objective_functions=objective_functions,
                                objective_functions_weights=[1, -1],
                                ga_params=gp_params_remote
                                )


    
if __name__ == '__main__':
    main()
