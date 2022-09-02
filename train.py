import datetime

import feature_engine.missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import tree     # Árvore de Decisão
from sklearn import ensemble # Random Forest
import sqlalchemy
import xgboost as xgb       

def train_rl(imputer_zero, imputer_um, onehot, X_train, y_train):
    ## Definição do modelo
    model_rl = linear_model.LogisticRegression(penalty='l1', 
                                               solver='liblinear' ) 

    full_pipeline_rl = Pipeline( steps=[('zero', imputer_zero),
                                        ('um', imputer_um),
                                        ("onehot", onehot),
                                        ('model', model_rl) ] )
    ## inspace
    param_grid = { 'model__C':np.linspace(0,0.22,5), 
                'model__random_state':[1992]}

    search_rl = model_selection.GridSearchCV(full_pipeline_rl,
                                            param_grid,
                                            cv=3,
                                            n_jobs=-1,
                                            scoring='roc_auc')
    ## Executa o treinamento!!
    search_rl.fit(X_train, y_train) 
    return search_rl.best_estimator_

def train_tree(imputer_zero, imputer_um, onehot, X_train, y_train):
    ## Definição do modelo
    model_tree = tree.DecisionTreeClassifier() 

    full_pipeline_tree = Pipeline( steps=[ ('zero', imputer_zero),
                                        ('um', imputer_um),
                                        ("onehot", onehot),
                                        ('modelo', model_tree) ] )

    param_grid = { "modelo__max_depth":[None, 5,6,7],
                "modelo__min_samples_split":[2,5],
                "modelo__min_samples_leaf":[1,2,5] }
    ## Declaração
    search_tree = model_selection.GridSearchCV(full_pipeline_tree,
                                param_grid,
                                cv=3,
                                n_jobs=-1,
                                scoring='neg_root_mean_squared_error') 
    ## Executa o treinamento!!
    search_tree.fit(X_train, y_train) 
    return search_tree.best_estimator_

def train_forest(imputer_zero, imputer_um, onehot, X_train, y_train):
    ## Definição do modelo
    model_rf = ensemble.RandomForestClassifier(random_state=1992) 

    full_pipeline_rf = Pipeline( steps=[('zero', imputer_zero),
                                        ('um', imputer_um),
                                        ('onehot', onehot),
                                        ('modelo', model_rf)])

    param_grid = { "modelo__n_estimators":[10,20],
                "modelo__max_depth":[5,10],
                "modelo__min_samples_split":[10,12],
                "modelo__min_samples_leaf":[5,10] }
    ## Declaração
    search_rf = model_selection.GridSearchCV(full_pipeline_rf,
                                param_grid,
                                cv=3,
                                n_jobs=-1,
                                scoring='neg_root_mean_squared_error') 
    ## Executa o treinamento!!
    search_rf.fit(X_train, y_train) 
    return search_rf.best_estimator_

def train_xgb(steps, X_train, y_train):
    model_xgb = xgb.XGBClassifier(random_state=1992)

    steps = steps + [('modelo', model_xgb)]
    full_pipeline_xgb = Pipeline(steps)

    param_grid = { "modelo__n_estimators":[10,50],
                "modelo__max_depth":[5,10],
                "modelo__eta":[0.1, 0.3],
                "modelo__subsample":[0.1, 0.2] }
    ## Declaração
    search_xgb = model_selection.GridSearchCV(full_pipeline_xgb,
                                param_grid,
                                cv=3,
                                n_jobs=-1,
                                scoring='neg_root_mean_squared_error') 
    ## Executa o treinamento!!
    search_xgb.fit(X_train, y_train) 
    return search_xgb.best_estimator_

def train_select_model(DATA_PATH, MODEL_PATH):
    ''' Function to train all models, assess results and choose be one.
    ''' 

    con = sqlalchemy.create_engine( "sqlite:///" + DATA_PATH )
    df = pd.read_sql_table( "TB_ABT", con )

    target = 'fl_venda'
    to_remove = (['seller_id', 'seller_city', 'seller_zip_code_prefix', 
                 'dt_ref'] + [target])
    cat_vars = ['seller_state']
    num_vars = list(set(df.columns.tolist()) 
                        - set( to_remove ) 
                        - set(cat_vars))

    tend_var = [i for i in num_vars if i.startswith("tend")]
    qtd_vars = ([i for i in num_vars if i.startswith("qtd") 
                                     or i.startswith("quant")])
    media_vars = [i for i in num_vars if i.startswith("media")]
    max_vars = [i for i in num_vars if i.startswith("max")]
    dias = [i for i in num_vars if i.startswith("dias")]
    prop = [i for i in num_vars if i.startswith("prop")]

    # Convertendo para tipo datetime
    df['dt_ref'] = pd.to_datetime(df['dt_ref']) 
    # Pegando a data máxima
    max_dt = df['dt_ref'].max() 
    # Separando para base com maior datas
    df_oot = df[df['dt_ref']==max_dt].copy() 
    # Separando para base menor que a maior data
    df_train = df[df['dt_ref'] < max_dt].copy() 

    X_train, X_test, y_train, y_test = model_selection.train_test_split( 
                                            df_train[num_vars+cat_vars],
                                            df_train[target],
                                            random_state=1992,
                                            test_size=0.25)

    imputer_zero = mdi.ArbitraryNumberImputer( arbitrary_number=0,
                    variables=qtd_vars+media_vars+max_vars+dias+prop)

    imputer_um = mdi.ArbitraryNumberImputer( arbitrary_number=1,
                                            variables=tend_var)

    onehot = ce.OneHotCategoricalEncoder(variables=cat_vars, 
                                         drop_last=True) # Cria Dummys   

    # Regressão Logística
    best_model_rl = train_rl(imputer_zero, 
                            imputer_um, 
                            onehot, 
                            X_train, 
                            y_train)
    # Árvore de decisão
    best_model_tree = train_tree(imputer_zero, 
                                imputer_um, 
                                onehot, 
                                X_train, 
                                y_train)
    # Random Forest
    best_model_rf = train_forest(imputer_zero, 
                                imputer_um, 
                                onehot, 
                                X_train, 
                                y_train)

    # XGBoot 1
    steps = [('zero', imputer_zero), 
            ('um', imputer_um), ('onehot', onehot)]
    best_model_xgb = train_xgb(steps, X_train, y_train)

    # XGboot 2
    steps = [('onehot', onehot)] 
    champion_model = train_xgb(steps, X_train, y_train)

    # Checando performance
    # Verificando erro na base de teste
    y_test_prob_rl = best_model_rl.predict_proba(X_test)[:, 1]
    y_test_prob_tree = best_model_tree.predict_proba(X_test)[:, 1]
    y_test_prob_rf = best_model_rf.predict_proba(X_test)[:, 1]
    y_test_prob_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
    y_test_prob_xgb_nt = champion_model.predict_proba(X_test)[:, 1]

    auc_test_rl = metrics.roc_auc_score( y_test, y_test_prob_rl)
    auc_test_tree = metrics.roc_auc_score( y_test, y_test_prob_tree)
    auc_test_rf = metrics.roc_auc_score( y_test, y_test_prob_rf)
    auc_test_xgb = metrics.roc_auc_score( y_test, y_test_prob_xgb)
    auc_test_xgb_nt = metrics.roc_auc_score( y_test, y_test_prob_xgb_nt)

    # Criado DataFrame com os modelos e resultados
    all_models = (
        pd.DataFrame(columns=['name', 'modelo', 'auc'],
            data=[['Regressão Logística', best_model_rl, auc_test_rl],
                ['Árvore', best_model_tree, auc_test_tree],
                ['Random Forest', best_model_rf, auc_test_rf],
                ['XGB', best_model_xgb, auc_test_xgb],
                ['XGB NT', champion_model, auc_test_xgb_nt]] ))

    ## extraindo o modelo campeão
    champion_model = (
    all_models.sort_values('auc', 
                             ascending=False).head(1)['modelo'].item())

    champion_auc = ( all_models.sort_values('auc', 
                             ascending=False).head(1)['auc'].item())                             

    # Retreinar para a base toda (Sem OOT)
    champion_model.fit( df_train[num_vars+cat_vars], 
                           df_train[target])

    # Verificando performance na base Out Of Time
    y_test_prob = (
    champion_model.predict_proba(df_oot[ num_vars+cat_vars ])[:, 1])
    auc_oot = metrics.roc_auc_score( df_oot[target], y_test_prob)

    # Retreinando o modelo para a base realmente completa
    champion_model.fit(df[num_vars+cat_vars], df[target])

    # salvando o modelos no disco
    features = (champion_model[:-1].transform( 
               df_train[ num_vars+cat_vars ] ).columns.tolist())
    try:
        features_importance = (
            pd.Series(champion_model[-1].feature_importances_, 
            index= features))
        features_importance.sort_values( ascending=False ).head(20)
    except AttributeError:
        features_importance = None

    model_s = pd.Series( {"cat_vars":cat_vars,
                      "num_vars":num_vars,
                      "fit_vars": X_train.columns.tolist(),
                      "model":champion_model,
                      "auc":{"test": champion_auc, "oot":auc_oot}} )

    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    model_s.to_pickle(MODEL_PATH + f'/chamption_model - {now}.pkl ')