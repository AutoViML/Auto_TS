import pandas as pd
import numpy as np
np.random.seed(99)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgbm
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import csv
import re
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_log_error, mean_squared_error,balanced_accuracy_score
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp
import time
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter, defaultdict
import pdb
#################  All these imports are needed for the pipeline #######
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin #gives fit_transform method for free
import pdb
from sklearn.base import TransformerMixin
from collections import defaultdict
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer
###################################################################################################
#############   This is where you import from other Auto_TS modules ############
from ..utils import My_LabelEncoder, My_LabelEncoder_Pipe
from ..utils import left_subtract
#################################################################################
def complex_XGBoost_model(X_train, y_train, X_test, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label', n_splits=5, 
                                num_boost_round=1000, verbose=-1):
    """
    This model is called complex because it handle multi-label, mulit-class datasets which XGBoost ordinarily cant.
    Just send in X_train, y_train and what you want to predict, X_test
    It will automatically split X_train into multiple folds (10) and train and predict each time on X_test.
    It will then use average (or use mode) to combine the results and give you a y_test.
    It will automatically detect modeltype as "Regression" or 'Classification'
    It will also add MultiOutputClassifier and MultiOutputRegressor to multi_label problems.
    The underlying estimators in all cases is XGB. So you get the best of both worlds.

    Inputs:
    ------------
    X_train: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    y_train: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
    log_y: default = False: If True, it means use the log of the target variable "y" to train and test.
    GPU_flag: if your machine has a GPU set this flag and it will use XGBoost GPU to speed up processing.
    scaler : default is StandardScaler(). But you can send in MinMaxScaler() as input to change it or any other scaler.
    enc_method: default is 'label' encoding. But you can choose 'glmm' as an alternative. But those are the only two.
    verbose: default = 0. Choosing 1 will give you lot more output.

    Outputs:
    ------------
    y_preds: Predicted values for your X_XGB_test dataframe.
        It has been averaged after repeatedly predicting on X_XGB_test. So likely to be better than one model.
    """
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    ####################################
    start_time = time.time()
    top_num = 10
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    else:
        targets = Y_XGB.columns.tolist()
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
    modeltype, _ = analyze_problem_type(Y_XGB, targets)
    columns =  X_XGB.columns
    ##### Now continue with scaler pre-processing ###########
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists()
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['tree_method'] = 'hist'
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    if GPU_exists:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        print('    Hyper Param Tuning XGBoost with GPU parameters. This will take time. Please be patient...')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Hyper Param Tuning XGBoost with CPU parameters. This will take time. Please be patient...')
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number

    #########  Now set the number of rows we need to tune hyper params ###
    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}
    random_search_flag =  True

      #### We need a small validation data set for hyper-param tuning #########################
    hyper_frac = 0.2
    #### now select a random sample from X_XGB ##
    
    if modeltype == 'Regression':
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999)
    else:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999, stratify = Y_XGB)
        
    ######  This step is needed for making sure y is transformed to log_y ####################
    if modeltype == 'Regression' and log_y:
            Y_train = np.log(Y_train)
            Y_valid = np.log(Y_valid)

    #### First convert test data into numeric using train data ###
    X_train, Y_train, X_valid, Y_valid, scaler = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    
    ######  Time to hyper-param tune model using randomizedsearchcv and partial train data #########
    num_boost_round = xgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                         multi_label, log_y, num_boost_round=num_boost_round)

    #### First convert test data into numeric using train data ###############################
    if not isinstance(X_XGB_test, str):
        x_train, y_train, x_test, _, _ = data_transform(X_XGB, Y_XGB, X_XGB_test, "",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    ######  Time to train the hyper-tuned model on full train data ##########################
    random_search_flag = False
    model = xgbm_model_fit(random_search_flag, x_train, y_train, x_test, "", modeltype,
                                multi_label, log_y, num_boost_round=num_boost_round)
    
    #############  Time to get feature importances based on full train data   ################
    
    if multi_label:
        for i,target_name in enumerate(targets):
            each_model = model.estimators_[i]
            imp_feats = dict(zip(x_train.columns, each_model.feature_importances_))
            importances = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].values
            important_features = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
            print('Top 10 features for {}: {}'.format(target_name, important_features))
    else: 
        imp_feats = model.get_score(fmap='', importance_type='gain')
        importances = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].values
        important_features = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
        print('Top 10 features:\n%s' %important_features[:top_num])
        #######  order this in the same order in which they were collected ######
        feature_importances = pd.DataFrame(importances,
                                           index = important_features,
                                            columns=['importance'])
    
    ######  Time to consolidate the predictions on test data ################################
    if not multi_label and not isinstance(X_XGB_test, str):
        x_test = xgb.DMatrix(x_test)
    if isinstance(X_XGB_test, str):
        print('No predictions since X_XGB_test is empty string. Returning...')
        return {}
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            if log_y:
                pred_xgbs = np.exp(model.predict(x_test))
            else:
                pred_xgbs = model.predict(x_test)
            #### if there is no test data just return empty strings ###
        else:
            pred_xgbs = []
    else:
        if multi_label:
            pred_xgbs = model.predict(x_test)
            pred_probas = model.predict_proba(x_test)
        else:
            pred_probas = model.predict(x_test)
            if modeltype =='Multi_Classification':
                pred_xgbs = pred_probas.argmax(axis=1)
            else:
                pred_xgbs = (pred_probas>0.5).astype(int)
    ##### once the entire model is trained on full train data ##################
    print('    Time taken for training XGBoost on entire train data (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if multi_label:
        for i,target_name in enumerate(targets):
            each_model = model.estimators_[i]
            xgb.plot_importance(each_model, importance_type='gain', title='XGBoost model feature importances for %s' %target_name)
    else:
        xgb.plot_importance(model, importance_type='gain', title='XGBoost final model feature importances')
    print('Returning the following:')
    print('    Model = %s' %model)
    print('    Scaler = %s' %scaler)    
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    (3) sample predictions:%s' %pred_xgbs[:3])
        return (pred_xgbs, scaler, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    (3) sample predictions (may need to be transformed to original labels):%s' %pred_xgbs[:3])
            print('    (3) sample predicted probabilities:%s' %pred_probas[:3])
        return (pred_xgbs, scaler, model)
##############################################################################################
import xgboost as xgb
def xgbm_model_fit(random_search_flag, x_train, y_train, x_test, y_test, modeltype,
                         multi_label, log_y, num_boost_round=100):
    start_time = time.time()
    if multi_label and not random_search_flag:
        model = num_boost_round
    else:
        rand_params = {
            'learning_rate': sp.stats.uniform(scale=1),
            'gamma': sp.stats.randint(0, 100),
            'n_estimators': sp.stats.randint(100,500),
            "max_depth": sp.stats.randint(3, 15),
                }

    if modeltype == 'Regression':
        objective = 'reg:squarederror' 
        eval_metric = 'rmse'
        shuffle = False
        stratified = False
        num_class = 0
        score_name = 'Score'
        scale_pos_weight = 1
    else:
        if modeltype =='Binary_Classification':
            objective='binary:logistic'
            eval_metric = 'error' ## dont foolishly change to auc or aucpr since it doesnt work in finding feature imps later
            shuffle = True
            stratified = True
            num_class = 1
            score_name = 'Error Rate'
            scale_pos_weight = get_scale_pos_weight(y_train)
        else:
            objective = 'multi:softprob'
            eval_metric = 'merror'  ## dont foolishly change to auc or aucpr since it doesnt work in finding feature imps later
            shuffle = True
            stratified = True
            if multi_label:
                num_class = y_train.nunique().max() 
            else:
                if isinstance(y_train, np.ndarray):
                    num_class = np.unique(y_train).max() + 1
                elif isinstance(y_train, pd.Series):
                    num_class = y_train.nunique()
                else:
                    num_class = y_train.nunique().max() 
            score_name = 'Multiclass Error Rate'
            scale_pos_weight = 1  ### use sample_weights in multi-class settings ##
    ######################################################
    final_params = {
          'booster' :'gbtree',
          'colsample_bytree': 0.5,
          'alpha': 0.015,
          'gamma': 4,
          'learning_rate': 0.01,
          'max_depth': 8,
          'min_child_weight': 2,
          'reg_lambda': 0.5,
          'subsample': 0.7,
          'random_state': 99,
          'objective': objective,
          'eval_metric': eval_metric,
          'verbosity': 0,
          'n_jobs': -1,
          'scale_pos_weight':scale_pos_weight,
          'num_class': num_class,
          'silent': True
            }
    #######  This is where we split into single and multi label ############
    if multi_label:
        ######   This is for Multi_Label problems ############
        rand_params = {'estimator__learning_rate':[0.1, 0.5, 0.01, 0.05],
          'estimator__n_estimators':[50, 100, 150, 200, 250],
          'estimator__gamma':[2, 4, 8, 16, 32],
          'estimator__max_depth':[3, 5, 8, 12],
          }
        if random_search_flag:
            if modeltype == 'Regression':
                clf = XGBRegressor(n_jobs=-1, random_state=999, max_depth=6)
                clf.set_params(**final_params)
                model = MultiOutputRegressor(clf, n_jobs=-1)
            else:
                clf = XGBClassifier(n_jobs=-1, random_state=999, max_depth=6)
                clf.set_params(**final_params)
                model = MultiOutputClassifier(clf, n_jobs=-1)
            if modeltype == 'Regression':
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'precision'
            model = RandomizedSearchCV(model,
                       param_distributions = rand_params,
                       n_iter = 15,
                       return_train_score = True,
                       random_state = 99,
                       n_jobs=-1,
                       cv = 3,
                       refit=True,
                       scoring = scoring,
                       verbose = False)        
            model.fit(x_train, y_train)
            print('Time taken for Hyper Param tuning of multi_label XGBoost (in minutes) = %0.1f' %(
                                            (time.time()-start_time)/60))
            cv_results = pd.DataFrame(model.cv_results_)
            print('Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
            ### In this case, there is no boost rounds so just return the default num_boost_round
            return model.best_estimator_
        else:
            try:
                model.fit(x_train, y_train)
            except:
                print('Multi_label XGBoost model is crashing during training. Please check your inputs and try again...')
            return model
    else:
        #### This is for Single Label Problems #############
        if modeltype == 'Multi_Classification':
            wt_array = get_sample_weight_array(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=wt_array)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        ########   Now let's perform randomized search to find best hyper parameters ######
        if random_search_flag:
            cv_results = xgb.cv(final_params, dtrain, num_boost_round=num_boost_round, nfold=5, 
                stratified=stratified, metrics=eval_metric, early_stopping_rounds=10, seed=999, shuffle=shuffle)
            # Update best eval_metric
            best_eval = 'test-'+eval_metric+'-mean'
            mean_mae = cv_results[best_eval].min()
            boost_rounds = cv_results[best_eval].argmin()
            print("Cross-validated %s = %0.3f in num rounds = %s" %(score_name, mean_mae, boost_rounds))
            print('Time taken for Hyper Param tuning of XGBoost (in minutes) = %0.1f' %(
                                                (time.time()-start_time)/60))
            return boost_rounds
        else:
            try:
                model = xgb.train(
                    final_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    verbose_eval=False,
                )
            except:
                print('XGBoost model is crashing. Please check your inputs and try again...')
            return model
####################################################################################
# Calculate class weight
from sklearn.utils.class_weight import compute_class_weight
import copy
from collections import Counter
def find_rare_class(classes, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    counts = OrderedDict(Counter(classes))
    total = sum(counts.values())
    if verbose >= 1:
        print('       Class  -> Counts -> Percent')
        sorted_keys = sorted(counts.keys())
        for cls in sorted_keys:
            print("%12s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
    if type(pd.Series(counts).idxmin())==str:
        return pd.Series(counts).idxmin()
    else:
        return int(pd.Series(counts).idxmin())
###################################################################################
def get_sample_weight_array(y_train):
    y_train = copy.deepcopy(y_train)    
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    elif isinstance(y_train, pd.Series):
        pass
    elif isinstance(y_train, pd.DataFrame):
        ### if it is a dataframe, return only if it s one column dataframe ##
        y_train = y_train.iloc[:,0]
    else:
        ### if you cannot detect the type or if it is a multi-column dataframe, ignore it
        return None
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    if len(class_weights[(class_weights < 1)]) > 0:
        ### if the weights are less than 1, then divide them until the lowest weight is 1.
        class_weights = class_weights/min(class_weights)
    else:
        class_weights = (class_weights)
    ### even after you change weights if they are all below 1.5 do this ##
    #if (class_weights<=1.5).all():
    #    class_weights = np.around(class_weights+0.49)
    class_weights = class_weights.astype(int)    
    wt = dict(zip(classes, class_weights))

    ### Map class weights to corresponding target class values
    ### You have to make sure class labels have range (0, n_classes-1)
    wt_array = y_train.map(wt)
    #set(zip(y_train, wt_array))

    # Convert wt series to wt array
    wt_array = wt_array.values
    return wt_array
###############################################################################
from collections import OrderedDict
def get_scale_pos_weight(y_input):    
    y_input = copy.deepcopy(y_input)
    if isinstance(y_input, np.ndarray):
        y_input = pd.Series(y_input)
    elif isinstance(y_input, pd.Series):
        pass
    elif isinstance(y_input, pd.DataFrame):
        ### if it is a dataframe, return only if it s one column dataframe ##
        y_input = y_input.iloc[:,0]
    else:
        ### if you cannot detect the type or if it is a multi-column dataframe, ignore it
        return None
    classes = np.unique(y_input)
    rare_class = find_rare_class(y_input)
    xp = Counter(y_input)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_input)
    
    if len(class_weights[(class_weights < 1)]) > 0:
        ### if the weights are less than 1, then divide them until the lowest weight is 1.
        class_weights = class_weights/min(class_weights)
    else:
        class_weights = (class_weights)
    ### even after you change weights if they are all below 1.5 do this ##
    #if (class_weights<=1.5).all():
    #    class_weights = np.around(class_weights+0.49)

    class_weights = class_weights.astype(int)
    class_weights[(class_weights<1)]=1
    class_rows = class_weights*[xp[x] for x in classes]
    class_rows = class_rows.astype(int)
    class_weighted_rows = dict(zip(classes,class_weights))
    rare_class_weight = class_weighted_rows[rare_class]
    print('    For class %s, weight = %s' %(rare_class, rare_class_weight))
    return rare_class_weight
#########################################################################################################
###########################################################################################
from collections import defaultdict
from collections import OrderedDict
from sklearn.impute import SimpleImputer
def data_transform(X_train, Y_train, X_test="", Y_test="", modeltype='Classification',
            multi_label=False, enc_method='label', scaler=""):
    
    #### All these are needed for transforming cat variables and building a pipeline ###
    imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
    ohe = OneHotEncoder()
    imp_ohe = make_pipeline(imp_constant, ohe)
    vect = CountVectorizer()
    imp = SimpleImputer()
    le = My_LabelEncoder()

    def drop_second_col(Xt):
        ### This deletes the 2nd column. Hence col number=1 and axis=1 ###
        return np.delete(Xt, 1, 1)

    ####  This is where we define the Pipeline for cat encoders and Label Encoder ############
    lep = My_LabelEncoder_Pipe()
    drop_second_col_func = FunctionTransformer(drop_second_col)
    ### lep_one uses My_LabelEncoder to first label encode and then drop the second unused column ##
    lep_one = make_pipeline(lep, drop_second_col_func)

    ### if you drop remainder variables, then leftovervars is not needed.
    ### If you passthrough remainder variables, then leftovers must be included 
    remainder = 'drop'

    ### If you choose MaxAbsScaler, then NaNs which were Label Encoded as -1 are preserved as - (negatives). This is fantastic.
    ### If you choose StandardScaler or MinMaxScaler, the integer values become stretched as if they are far 
    ###    apart when in reality they are close. So avoid it for now.
    scaler = MaxAbsScaler()
    #scaler = StandardScaler()
    ##### First make sure that the originals are not modified ##########
    X_train_encoded = copy.deepcopy(X_train)
    X_test_encoded = copy.deepcopy(X_test)
    ##### Use My_Label_Encoder to transform label targets if needed #####
    if multi_label:
        if modeltype != 'Regression':
            targets = Y_train.columns
            Y_train_encoded = copy.deepcopy(Y_train)
            for each_target in targets:
                mlb = My_LabelEncoder()
                if not isinstance(Y_train, str):
                    Y_train_encoded[each_target] = mlb.fit_transform(Y_train[each_target])
                else:
                    Y_train_encoded = copy.deepcopy(Y_train)
                if not isinstance(Y_test, str):
                    Y_test_encoded= mlb.transform(Y_test)
                else:
                    Y_test_encoded = copy.deepcopy(Y_test)
        else:
            Y_train_encoded = copy.deepcopy(Y_train)
            Y_test_encoded = copy.deepcopy(Y_test)
    else:
        if modeltype != 'Regression':
            mlb = My_LabelEncoder()
            if not isinstance(Y_train, str):
                Y_train_encoded= mlb.fit_transform(Y_train)
            else:
                Y_train_encoded = copy.deepcopy(Y_train)
            if not isinstance(Y_test, str):
                Y_test_encoded= mlb.transform(Y_test)
            else:
                Y_test_encoded = copy.deepcopy(Y_test)
        else:
            Y_train_encoded = copy.deepcopy(Y_train)
            Y_test_encoded = copy.deepcopy(Y_test)
    
    #### This is where we find out how to transform X_train and X_test  ####
    catvars = X_train.select_dtypes('object').columns.tolist() + X_train.select_dtypes('category').columns.tolist()
    numvars = X_train.select_dtypes('number').columns.tolist()
    ########    This is where we define the pipeline for cat variables ###########
    ### How do we make sure that we create one new LE_Pipe for each catvar? This is one way.
    init_str = 'make_column_transformer('
    middle_str = "".join(['(lep_one, catvars['+str(i)+']),' for i in range(len(catvars))])
    end_str = '(imp, numvars),    remainder=remainder)'
    full_str = init_str+middle_str+end_str
    ct = eval(full_str)
    pipe = make_pipeline(ct, scaler )
    ###  You will get a multidimensional numpy array ############
    dfo = pipe.fit_transform(X_train)
    if not isinstance(X_test, str):
        dfn = pipe.fit_transform(X_test)
    
    ### The first columns should be whatever is in the Transformer_Pipeline list of columns
    ### Hence they will be catvars. The second list will be numvars. Then only other columns that are passed through.
    ### So after the above 2 lists, you will get remainder cols unchanged: we call them leftovers.
    leftovervars = left_subtract(X_train.columns.tolist(), catvars+numvars)

    ## So if you do it correctly, you will get the list of names in proper order this way:
    ## first is catvars, then numvars and then leftovervars
    if remainder == 'drop':
        cols_names = catvars+numvars                    
    else:
        cols_names = catvars+numvars+leftovervars

    dfo = pd.DataFrame(dfo, columns = cols_names)
    if not isinstance(X_test, str):
        dfn = pd.DataFrame(dfn, columns = cols_names)
    
    copy_names = copy.deepcopy(cols_names)

    for each_col in copy_names:
        X_train_encoded[each_col] = dfo[each_col].values
        if not isinstance(X_test, str):
            X_test_encoded[each_col] = dfn[each_col].values

    return X_train_encoded, Y_train_encoded, X_test_encoded, Y_test_encoded, pipe
##################################################################################
def analyze_problem_type(train, target, verbose=0) :
    """
    ##################################################################################
    ########## Analyze if it is a Regression or Classification type problem    #######
    ##################################################################################
    """
    target = copy.deepcopy(target)
    train = copy.deepcopy(train)
    if isinstance(train, pd.Series):
        train = pd.DataFrame(train)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    if isinstance(target, str):
        target = [target]
    if len(target) == 1:
        targ = target[0]
        multilabel = False
    else:
        targ = target[0]
        multilabel = True
    ####  This is where you detect what kind of problem it is #################
    if  train[targ].dtype in ['int64', 'int32','int16']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float16','float32','float64']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    else:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose >= 1:
        if multilabel:
            print('''\n########### Multi-Label %s Model Tuning and Training Started  ####''' %(model_class))
        else:
            print('''\n########### Single-Label %s Model Tuning and Training Started ####''' %(model_class))
    return model_class, multilabel
#############################################################################
