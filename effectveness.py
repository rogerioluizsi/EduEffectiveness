
def mode_high_frequencie(df):
    columns_dropped = []
    ammount=0
    before= df.shape[1]
    #print("Number of features: ", before)
    for i in df:
        mode = df[i].mode()[0]
        threshold = 0.9
        count = df[(df[i]== mode)].shape[0]
        freq = count/df.shape[0]
        if freq >= threshold:
            ammount +=1
            #print("drop out", [i], "mode = ", mode )
            #df.drop([i], inplace = True, axis=1)
            columns_dropped.append(i)
            
    #print("Total Dropped: ",ammount)
    #print ("Remainning: ", before-ammount)  
    return(columns_dropped)

def clip_tail(df):
    quantitative = df[(df.nunique() > 2).index[(df.nunique() > 2)]].columns.to_list()
    #print(len(quantitative)
    df[quantitative]=df[quantitative].apply(lambda x: x.clip(upper = (np.percentile(x, 97.5))))
    df[quantitative]=df[quantitative].apply(lambda x: x.clip(lower = (np.percentile(x, 2.5))))
    return(df)

def scaler (df):
    scaler = MinMaxScaler()
    x = df.values
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns = df.columns)
    return (df)

def build_target(df):
    df['TARGET'] = pd.qcut (df.NU_NOTA_GERAL, 4, labels = [1,2,3,4]).map(lambda x : 0 if x<3 else 1) 
    #rint((df.TARGET==0).sum()/(df.TARGET.count())*100, '% lowers quartis')
    return df['TARGET']

def abroca(df, group_ref, baseline):
    #print(group_ref)
    pred_res =pd.DataFrame([], columns= [political_unit, 'abroca'])

    for i in df[group_ref].unique():
        if i != baseline:
            #print(i, 'aqui', baseline)
            temp= df[['y_proba' , 'y_true', political_unit]]
            temp = temp[(temp[group_ref]== baseline) | (temp[group_ref]==i)]
        #temp = pd.get_dummies(temp)
            temp[political_unit] = LabelEncoder().fit_transform(temp[political_unit])
        #print(temp.head())
            prot_attr = temp.iloc[:,2].name
        #print(prot_attr)
        #temp.iloc[:,2] =temp.iloc[:,2].astype('str')
       
            abroca_score = compute_abroca(temp, pred_col = 'y_proba' , label_col = 'y_true', n_grid = 10000,
                                majority_protected_attr_val = 1,compare_type='binary',protected_attr_col = prot_attr, plot_slices =False)
        #print(abroca_score)
    
            res = pd.DataFrame({political_unit: [i], 'abroca': list(abroca_score.values())})
            pred_res = pred_res.append(res)
    return(pred_res)

def compute_scores (pred_res, political_unit, ref):
    groups = [political_unit] + ['model']
    #.groupby(groups).apply(lambda x: abroca(x, political_unit, ref))
    #print(pred_res)
    #pred_res['auc_general'] = pred_res.groupby('model').apply(lambda x: roc_auc_score(x.y_true, x.y_proba))
    #print(pred_res['auc_general'])
    #pred_res['f1_general'] = pred_res.groupby('model').apply(lambda x: f1_score(x.y_true.to_list(), x.y_pred.to_list()))
    #print(temp)
    
    #print (groups)
    pred_eval_score_general = pred_res.groupby('model').apply(lambda x: pd.DataFrame(
            {
                'auc_model': [roc_auc_score(x.y_true, x.y_proba)],
                'f1_model': f1_score(x.y_true.to_list(), x.y_pred.to_list())
            }
            )).droplevel(1).apply(pd.to_numeric)  
    
    pred_eval_score = pred_res.groupby(groups).apply(lambda x: pd.DataFrame(
            {
                'tp': [len(x.query('y_true == 1 and y_pred == 1'))],
                'tn': [len(x.query('y_true == 0 and y_pred == 0'))],
                'fp': [len(x.query('y_true == 0 and y_pred == 1'))],
                'fn': [len(x.query('y_true == 1 and y_pred == 0'))],
                'auc':(roc_auc_score(x.y_true, x.y_proba) if len(np.unique(x.y_true)) != 1 else 0) 
               
            
            }
            )).droplevel(2).apply(pd.to_numeric)
    #print('first',pred_eval_score.head())

    pred_eval_score = pred_eval_score.assign(f1=lambda x: 2*x['tp']/(2*x['tp']+x['fp']+x['fn']))      
    pred_eval_score = pred_eval_score.assign(acc=lambda x: (x['tp']+x['tn'])/(x['tp']+x['tn']+x['fp']+x['fn']))
    pred_eval_score = pred_eval_score.assign(fnr=lambda x: x['fn']/(x['fn']+x['tp']))
    pred_eval_score = pred_eval_score.assign(fpr=lambda x: x['fp']/(x['fp']+x['tn']))
    #pred_eval_score = pred_eval_score.assign(auc=auc)
    pred_eval_score = pred_eval_score.reset_index()
    #print('primeiro',pred_eval_score.head())
    #print('segundo',pred_eval_score_general.head())
    pred_eval_score = pd.merge(pred_eval_score, pred_eval_score_general, on ='model', how='left')
    #print('final',pred_eval_score.head())
    return (pred_eval_score)

def df_proportion_class (DF, tp_escola, level):
        level = 'codigo_uf' if level == 'estado' else 'regiao'
        df, df2 = pd.DataFrame(), pd.DataFrame()
        mun_est = DF[(DF.IN_TP_ESCOLA == tp_escola)]
    
        for i in mun_est.CO_ANO.unique():  
            mun_est_y = mun_est[mun_est['CO_ANO']==i]
            mun_est_y['target']= build_target(mun_est_y).to_numpy()
            grp = mun_est_y[['CO_ANO',level, 'target']].groupby([level])
            df[i] = grp.apply (lambda x: x.target.sum()/x.target.count())
            df2[i] = mun_est_y[['CO_ANO',level, 'NU_NOTA_GERAL']].groupby([level])['NU_NOTA_GERAL'].mean()
        return df, df2

def build_df (level, metric, big_table, year):
    level = 'codigo_uf' if level == 'estado' else 'regiao'
    df = pd.DataFrame()
    for k, v in big_table.items():
        #print(k)
        if (k == year):
            for nk, nv in v.items(): 
                #print(nk)
                nv = nv.groupby(level).apply(lambda x: x.loc[x.f1_model == x.f1_model.max()]).droplevel(0)
                for i in range (nv.shape[0]):
                #print(i)
                #nv = nv.groupby('regiao')['auc'].max()
                    column = nv.reset_index()[level][i]
                    value = nv[[level,metric]].iloc[i][1]
                    model = nv.reset_index().model[i]
                    f1_model = nv.reset_index().f1_model[i]
                    auc_model = nv.reset_index().auc_model[i]
                    df.loc[nk, 'Model'] = model
                    df.loc[nk, 'F1_measure'] = f1_model
                    df.loc[nk, 'AUC'] = auc_model
                    df.loc[nk, column] = value
                
                #print(column, value)
                

    return (df.round(2))

 def run_models(data, group_ref, m, tp_school):

    global student
    global school
    global teacher
    global non_actionable
    groups ={}
    scores = {}
    data = data[data.IN_TP_ESCOLA == tp_school] # Filter only target type school
    group_var = [group_ref]
    logo = LeaveOneGroupOut()
    for year in data.CO_ANO.unique():
        data_y = data[data.CO_ANO==year]
        y = build_target(data_y).to_numpy()
        df= data_y[independents + group_var] #Get all features set 
        bad_columns = mode_high_frequencie(df[independents])
    # update each group list, removing them
        full = df[independents].loc[:, ~df[independents].columns.isin(bad_columns)].columns.to_list()  
        student = df[student].loc[:, ~df[student].columns.isin(bad_columns)].columns.to_list()
        school = df[school].loc[:, ~df[school].columns.isin(bad_columns)].columns.to_list()   
        teacher = df[teacher].loc[:, ~df[teacher].columns.isin(bad_columns)].columns.to_list()   
        non_actionable = df[non_actionable].loc[:, ~df[non_actionable].columns.isin(bad_columns)].columns.to_list()
        
    # create dictionry with all combinations of features    
        groups['FULL'] = df[full]
    #single variable
        groups['SCHOOL'] = df[school]
        groups['STUDENT'] = df[student]
        groups['TEACHER'] = df[teacher]
        groups['NON_ACTIONABLE'] = df[non_actionable]
    
    #two variables
        groups['SCHOOL_STUDENT'] = df[school + student]
        groups['SCHOOL_TEACHER'] = df[school + teacher ]
        groups['SCHOOL_NON_ACTIONABLE'] = df[school + non_actionable]
        groups['STUDENT_TEACHER'] = df[student+ teacher]
        groups['STUDENT_NON_ACTIONABLE'] = df[student + non_actionable]
        groups['TEACHER_NON_ACTIONABLE'] = df[teacher + non_actionable]
    #Three variables
        groups['SCHOOL_STUDENT_TEACHER'] = df[school + student + teacher]
        groups['SCHOOL_STUDENT_NON_ACTIONABLE'] = df[school + student + non_actionable]
        groups['SCHOOL_TEACHER_NON_ACTIONABLE'] = df[school + teacher+ non_actionable]
        groups['STUDENT_TEACHER_NON_ACTIONABLE'] = df[student + teacher + non_actionable]
        
        df = df.set_index(group_ref) 
        unique_id = df.index.to_frame().reset_index(drop=True)
        
        scores[year]= {}
        for gp in groups:
            pred_res = pd.DataFrame([], columns=unique_id.columns.tolist()+['y_true', 'y_pred', 'y_proba', 'model'])  
            scores[year][gp]= {}
            X = groups[gp].to_numpy()
            group = df.index.get_level_values(group_ref)
            #print(group)
            for m in models:
                name_classifier = type(m).__name__
                #print(name_classifier)
                ypred = cross_val_predict(m, X, y, cv=logo,groups=group)
                yproba = cross_val_predict(m, X, y, cv=logo,groups=group, method='predict_proba')[:,1]
            
                #ypred = cross_val_predict(m, X, y, cv=10)
                #yproba = cross_val_predict(m, X, y, cv=10,method='predict_proba')[:,1]
  
                res = pd.DataFrame({'y_true': y, 'y_pred': ypred, 'y_proba': yproba, 'model': name_classifier})
                pred_res = pred_res.append(pd.concat([unique_id, res], axis=1))
                pred_res['y_true'] = pred_res['y_true'].astype(int)
                #print(pred_res.head())
            
    
                scores[year][gp] = compute_scores(pred_res, political_unit, ref)
         
    
    return(scores,pred_res)





#### execute
rseed = 1234
models = [LogisticRegression(),  RandomForestClassifier(random_state=0), AdaBoostClassifier(random_state=0)]
political_unit = 'regiao'
#ref = 'PR'
tp_level = "Municipal+Estadual"
big_table, raw_pred = run_models(DATA, political_unit, models, tp_level)
     

