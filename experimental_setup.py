import time
import random
import os
import pandas as pd
import pickle
from functools import partial
import traceback
from sklearn.model_selection import train_test_split

import utils
import ga_nsga2

def loop_with_equal_evals2(ml_models, experiments, task_id_lists, base_save_folder, data_dir, num_runs, objective_functions, objective_functions_weights , ga_params):
    assert os.path.isdir(base_save_folder), f"Folder to save results does not exist: {base_save_folder}"
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for r in range(num_runs):
                for e, exp in enumerate(experiments):

                    save_folder = f"{base_save_folder}/{ml}/{taskid}_{r}_{exp}"
                    time.sleep(random.random()*5)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    else:
                        continue

                    print("working on ")
                    print(save_folder)

                    print("loading data")
                    super_seed = (m+t+r+e)*1000
                    print("Super Seed : ", super_seed)

                    # Split the data into training_validation and testing sets
                    X_train_val, y_train_val, X_test, y_test, features, sens_features = utils.load_task(data_dir, taskid, test_size=0.15, seed=r)

                    # Split the training set into training and validation sets
                    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=r)

                    print("starting ml")
                                    
                    try:  
                        print("Starting the fitting process. ")
                        if exp=='Equal Weights':
                            num_evals = ga_params['pop_size']*ga_params['max_gens']
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed', 'run', *objective_functions, *['train_'+k for k in objective_functions]])
                            for i in range(num_evals):
                                this_seed = super_seed + i
                                est = ml(random_state=this_seed)
                                est.fit(X_train, y_train)
                                print("Ending the fitting process. ")

                                train_score = utils.evaluate_objective_functions(est, X_val, y_val, objective_functions,sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = this_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)
                            hv_info = utils.calculate_hypervolume(scores, objective_functions)
                            with open(f"{save_folder}/hv_values.pkl", "wb") as f:
                                pickle.dump(hv_info, f)
                            
                            return

                        elif exp=='Deterministic Weights':
                            num_evals = ga_params['pop_size']*ga_params['max_gens']
                            weights =  utils.calc_weights(X_train, y_train, sens_features)
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed', 'run', *objective_functions, *['train_'+k for k in objective_functions]])
                            for i in range(num_evals):
                                this_seed = super_seed + i
                                est = ml(random_state=this_seed)
                                est.fit(X_train, y_train, weights)
                                print("Ending the fitting process. ")

                                train_score = utils.evaluate_objective_functions(est, X_val, y_val, objective_functions,sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = this_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)
                            hv_info = utils.calculate_hypervolume(scores, objective_functions)
                            with open(f"{save_folder}/hv_values.pkl", "wb") as f:
                                pickle.dump(hv_info, f)

                            return
                        
                        else:
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed', 'run', *objective_functions, *['train_'+k for k in objective_functions]])
                            ga_func = partial(utils.fitness_func_holdout, model = ml(random_state=super_seed), X_train=X_train, y_train=y_train, X_val =X_val, y_val=y_val, 
                                              sens_features=sens_features, objective_fuctions=objective_functions, objective_functions_weights=objective_functions_weights)
                            ga_func.__name__ = 'ga_func'
                            ga = ga_nsga2.GA(ind_size = 2**(len(sens_features)+ 1), random_state=super_seed, fitness_func= ga_func,**ga_params)
                            ga.optimize()

                            for j in range(ga.evaluated_individuals.shape[0]):
                                est = ml(random_state=super_seed)
                                weights = utils.partial_to_full_sample_weight(ga.evaluated_individuals.loc[j,'individual'], X_train, y_train, sens_features)
                                est.fit(X_train, y_train, weights)
                                print("Ending the fitting process. ")
                                
                                train_score = utils.evaluate_objective_functions(est, X_val, y_val, objective_functions,sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = super_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)
                            hv_info = utils.calculate_hypervolume(scores, objective_functions)
                            with open(f"{save_folder}/hv_values.pkl", "wb") as f:
                                pickle.dump(hv_info, f)

                            return
                    except Exception as e:
                        trace =  traceback.format_exc()
                        pipeline_failure_dict = {"taskid": taskid, "exp_name": exp,  "error": str(e), "trace": trace}
                        print("failed on ")
                        print(save_folder)
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

                        return
        
    print("all finished")
    