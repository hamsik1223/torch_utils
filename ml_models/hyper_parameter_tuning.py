from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
import xgboost as xgb

def optimise_param_xgb(params):
    
    print(params)
    p = {'learning_rate': params['learning_rate'],
         'max_depth': params['max_depth'], 
         'gamma': params['gamma'], 
         'min_child_weight': params['min_child_weight'], 
         'subsample': params['subsample'], 
         'colsample_bytree': params['colsample_bytree'], 
         'verbosity': 0, 
         'objective': 'reg:squarederror', ####binary:logistic
         'eval_metric': 'mae', 
         'tree_method': 'gpu_hist', 
         'random_state': 42,
        }
    
    scores = []
    gkf = PurgedGroupTimeSeriesSplit(n_splits = n_splits, group_gap = group_gap)
    for fold, (tr, te) in enumerate(gkf.split(X_df, y_df, fake_group_np)):
        X_tr, X_val = X_df.iloc[tr], X_df.iloc[te]
        y_tr, y_val = y_df.iloc[tr], y_df.iloc[te]
        d_tr = xgb.DMatrix(X_tr, y_tr)
        d_val = xgb.DMatrix(X_val, y_val)
        clf = xgb.train(p, d_tr, params['n_round'], [(d_val, 'eval')], verbose_eval = False)
        val_pred = clf.predict(d_val)
        score = mean_absolute_error(y_val, val_pred)
#         print(f'Fold {fold} ROC AUC:\t', score)
        scores.append(score)
        
        del clf, val_pred, d_tr, d_val, X_tr, X_val, y_tr, y_val, score
        rubbish = gc.collect()
        
    score_avg = weighted_average(scores)
    print(score_avg)
    return score_avg