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
def complex_XGBoost_model(x_train, y_train, x_test, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label', n_splits=5, verbose=0):
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
    X_XGB: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    Y_XGB: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_XGB_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
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
    X_XGB = copy.deepcopy(x_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(x_test)
    ####################################
    num_boost_round = 1000
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

    ######   T H I S    I S   F O R    M U L T I    L A B E L    P R O B L E M S ###########
    #### We need a small validation data set for hyper-param tuning #########################
    hyper_frac = 0.2
    #### now select a random sample from X_XGB ##
    if modeltype == 'Regression':
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999)
    else:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999, stratify = Y_XGB)
    
    #### First convert test data into numeric using train data ###
    X_train, Y_train, X_valid, Y_valid, scaler = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    ######  This step is needed for making sure y is transformed to log_y ####################
    if modeltype == 'Regression' and log_y:
            Y_train = np.log(Y_train)

    ######  Time to hyper-param tune model using randomizedsearchcv and partial train data #########
    num_boost_round = xgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                         multi_label, log_y, num_boost_round=num_boost_round, n_splits=n_splits)

    ### this is where you insert the cut code ###

    #### First convert test data into numeric using train data ###############################
    if not isinstance(X_XGB_test, str):
        x_train, y_train, x_test, _, _ = data_transform(X_XGB, Y_XGB, X_XGB_test, "",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    ######  Time to train the hyper-tuned model on full train data ##########################
    random_search_flag = False
    model = xgbm_model_fit(random_search_flag, x_train, y_train, x_test, "", modeltype,
                                multi_label, log_y, num_boost_round=num_boost_round, n_splits=n_splits)
    
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
            if not multi_label:
                ### turn it into a 1-d array otherwise it will error later ##
                pred_xgbs = pred_xgbs.reshape(-1,1)
        else:
            pred_xgbs = []
    else:
        #### This is for Classification problems ##########
        if multi_label:
            pred_xgbs = model.predict(x_test)
            pred_probas = model.predict_proba(x_test)
        else:
            pred_probas = model.predict(x_test)
            if modeltype == 'Binary_Classification':
                pred_xgbs = (pred_probas>0.5).astype(int)
            else:
                pred_xgbs = pred_probas.argmax(axis=1)
            if not multi_label:
                ### turn it into a 1-d array otherwise it will error later ##
                pred_xgbs = pred_xgbs.reshape(-1,1)
    ##### once the entire model is trained on full train data ##################
    print('    Time taken for training XGBoost on entire train data (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if verbose >= 2:
        if multi_label:
            for i,target_name in enumerate(targets):
                each_model = model.estimators_[i]
                xgb.plot_importance(each_model, title='XGBoost model feature importances for %s' %target_name)
        else:
            xgb.plot_importance(model, title='XGBoost final model feature importances')
    print('\nModel tuning and training complete. Returning the following:')
    print('    Model = %s' %model)
    
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
                         multi_label, log_y, num_boost_round=100, n_splits=5):
    start_time = time.time()
    if multi_label:
        rand_params = {
            'estimator__learning_rate':[0.01,0.5,0.1,0.3,0.5],
            'estimator__n_estimators':[50, 100, 150, 200, 250, 300, 350]
                }
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
        scoring = 'neg_mean_squared_error'
        score_name = 'MSE'
    else:
        if modeltype =='Binary_Classification':
            objective='binary:logistic'
            eval_metric = 'error' ## dont foolishly change to auc or aucpr since it doesnt work in finding feature imps later
            shuffle = True
            stratified = True
            num_class = 1
            scoring='precision'
            score_name = 'Error Rate'
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
            scoring='precision'
            score_name = 'Multi Class error rate'
    
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
          #grow_policy='lossguide',
          'num_class': num_class,
          'silent': True
            }
    #######  This is where we split into single and multi label ############
    if multi_label:
        ######   This is for Multi_Label problems ############
        if modeltype == 'Regression':
            clf = XGBRegressor(n_jobs=-1, random_state=999, max_depth=6)
            clf.set_params(**final_params)
            model = MultiOutputRegressor(clf)
        else:
            clf = XGBClassifier(n_jobs=-1, random_state=999, max_depth=6)
            clf.set_params(**final_params)
            model = MultiOutputClassifier(clf)
        if random_search_flag:
            rsv = RandomizedSearchCV(model,
                       param_distributions = rand_params,
                       n_iter = 5,
                       return_train_score = True,
                       random_state = 99,
                       n_jobs=-1,
                       scoring=scoring,
                       cv = n_splits,
                       verbose = False)        
            rsv.fit(x_train, y_train)
            print('Time taken for Hyper Param tuning of multi_label XGBoost (in minutes) = %0.1f' %(
                                            (time.time()-start_time)/60))
            cv_results = pd.DataFrame(rsv.cv_results_)
            cv_mean = cv_results['mean_test_score'].mean()
            if cv_mean < 0:
                print('Mean cross-validated test %s = %0.04f' %(score_name, -1*cv_mean))
            else:
                print('Mean cross-validated test %s = %0.04f' %(score_name, cv_mean))
            ### In this case, there is no boost rounds so just return the best estimator
            return rsv.best_estimator_
        else:
            ### if it is multi_label, you get the best estimator in num_boost_round
            model = num_boost_round
            try:
                model.fit(x_train, y_train)
            except:
                print('Multi_label XGBoost model is crashing during training. Please check your inputs and try again...')
            return model
    else:
        
        #### This is for Single Label Problems #############
        dtrain = xgb.DMatrix(x_train, label=y_train)
        ########   Now let's perform randomized search to find best hyper parameters ######
        if random_search_flag:
            cv_results = xgb.cv(final_params, dtrain, num_boost_round=num_boost_round, nfold=n_splits, 
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
#########################################################################################################
###########################################################################################
from collections import defaultdict
from collections import OrderedDict
from sklearn.impute import SimpleImputer
def data_transform(X_train, Y_train, X_test="", Y_test="", modeltype='Classification',
            multi_label=False, enc_method='label', scaler=""):
    if isinstance(scaler, str):
        scaler = StandardScaler()
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
    
    #### This is where we find out if test data is given ####
    ####### Set up feature to encode  ####################
    feature_to_encode = X_train.columns[X_train.dtypes == 'O'].tolist()
    #print('features to label encode: %s' %feature_to_encode)
    #### Do label encoding now #################
    if enc_method == 'label':
        for feat in feature_to_encode:
            # Initia the encoder model
            lbEncoder = My_LabelEncoder()
            # fit the train data
            lbEncoder.fit(X_train[feat])
            # transform training set
            X_train_encoded[feat] = lbEncoder.transform(X_train[feat])
            # transform test set
            if not isinstance(X_test_encoded, str):
                X_test_encoded[feat] = lbEncoder.transform(X_test[feat])
    else:
        print('No encoding transform performed')

    ### make sure there are no missing values ###
    try:
        imputer = SimpleImputer(strategy='constant', fill_value=0, verbose=0, add_indicator=True)
        imputer.fit_transform(X_train_encoded)
        if not isinstance(X_test_encoded, str):
            imputer.transform(X_test_encoded)
    except:
        X_train_encoded = X_train_encoded.fillna(0)
        if not isinstance(X_test_encoded, str):
            X_test_encoded = X_test_encoded.fillna(0)

    # fit the scaler to the entire train and transform the test set
    scaler.fit(X_train_encoded)
    # transform training set
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_encoded), 
        columns=X_train_encoded.columns, index=X_train_encoded.index)
    # transform test set
    if not isinstance(X_test_encoded, str):
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), 
            columns=X_test_encoded.columns, index=X_test_encoded.index)
    return X_train_scaled, Y_train_encoded, X_test_scaled, Y_test_encoded, scaler
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
            print('''\n###########      Multi-Label %s Model Tuning and Training Started         ####''' %(model_class))
        else:
            print('''\n###########      Single-Label %s Model Tuning and Training Started        ####''' %(model_class))
    return model_class, multilabel
#############################################################################
from sklearn.base import TransformerMixin
from collections import defaultdict
class My_LabelEncoder(TransformerMixin):
    """
    ################################################################################################
    ######  This Label Encoder class works just like sklearn's Label Encoder!  #####################
    #####  You can label encode any column in a data frame using this new class. But unlike sklearn,
    the beauty of this function is that it can take care of NaN's and unknown (future) values.
    It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.
    ################################################################################################
    ##################### This is the best working version - don't mess with it! ###################
    Usage:
          MLB = My_LabelEncoder()
          train[column] = MLB.fit_transform(train[column])
          test[column] = MLB.transform(test[column])
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)

    def fit(self,testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        outs = np.unique(testx.factorize()[0])
        ins = testx.value_counts(dropna=False).index
        #if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
        #    ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs.tolist()))
        self.inverse_transformer = dict(zip(outs.tolist(),ins))
        return self

    def transform(self, testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        ### now convert the input to transformer dictionary values
        ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                max_val = np.max(list(self.transformer.values())) + 1
                self.transformer[each_missing] = max_val
                self.inverse_transformer[max_val] = each_missing
        outs = testx.map(self.transformer).values
        testk = testx.map(self.transformer)
        if testx.dtype not in [np.int16, np.int32, np.int64, float, bool, object]:
            if testx.isnull().sum().sum() > 0:
                fillval = self.transformer[np.nan]
                testk = testk.cat.add_categories([fillval])
                testk = testk.fillna(fillval)
                testk = testk.astype(int)
                return testk
            else:
                testk = testk.astype(int)
                return testk
        else:
            return outs

    def inverse_transform(self, testx):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################