"""Run: python -m student_success.train. Writes a complete, reproducible experiment."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import json
import hashlib
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from .config import ROOT,DATA_PATH,ARTIFACT_DIR,REPORT_DIR,SEED,CLASS_NAMES,PROTOCOL
from .schema import FEATURES,ADMISSION_FEATURES,NUMERICAL,validate_features,schema_records,defaults
from .modeling import candidate_model,align_probabilities,evaluate,policy_metrics,bootstrap_intervals
from .inference import sha256_file

EXPECTED_DATA_SHA256='a37dbda5555089a8d39b8be6f1a242f68403963a1f81423115c9676c8c7100e9'

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def cross_validate_candidate(name,features,data,indices):
    x=data.iloc[indices][features];y=data.iloc[indices].Status.to_numpy()
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    fold_rows=[];oof=np.zeros((len(y),3))
    base=DummyClassifier(strategy='prior') if name=='dummy' else candidate_model(name,features)
    for fold,(tr,va) in enumerate(cv.split(x,y),1):
        t=time.time();model=clone(base)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x.iloc[tr],y[tr])
        p=align_probabilities(model,x.iloc[va]);oof[va]=p
        m=evaluate(y[va],p)
        fold_rows.append({'candidate':name,'feature_set':'semester1' if features==FEATURES else 'admission',
            'fold':fold,'macro_f1':m['macro_f1'],'accuracy':m['accuracy'],
            'dropout_brier':m['dropout_brier'],'dropout_average_precision':m['dropout_average_precision'],
            'seconds':round(time.time()-t,3)})
        print(f'{name} / {len(features)} features / fold {fold}: macro F1 {m["macro_f1"]:.4f}',flush=True)
    return fold_rows,oof

def main():
    ARTIFACT_DIR.mkdir(exist_ok=True);REPORT_DIR.mkdir(exist_ok=True)
    data=pd.read_csv(DATA_PATH,sep=';')
    if sha256_file(DATA_PATH)!=EXPECTED_DATA_SHA256:
        raise ValueError('Dataset checksum changed. Version the data and protocol before a new experiment.')
    if set(data.Status)!=set(CLASS_NAMES):raise ValueError('Unexpected target labels')
    validate_features(data)
    all_idx=np.arange(len(data))
    development,holdout=train_test_split(all_idx,test_size=.2,random_state=SEED,stratify=data.Status)
    train,policy=train_test_split(development,test_size=.25,random_state=SEED+1,stratify=data.iloc[development].Status)
    split=np.full(len(data),'historical_holdout',dtype=object)
    split[train]='model_development';split[policy]='policy_validation'
    pd.DataFrame({'source_row':all_idx+1,'split':split,'target':data.Status}).to_csv(REPORT_DIR/'split_assignments.csv',index=False)
    write_json(REPORT_DIR/'protocol.json',PROTOCOL)
    folds=[];candidate_summaries=[];oof_lookup={}
    for name in ['dummy','logistic','random_forest','hist_gradient_boosting']:
        rows,oof=cross_validate_candidate(name,FEATURES,data,train)
        folds.extend(rows);oof_lookup[name]=oof
        a=np.array([r['macro_f1'] for r in rows])
        candidate_summaries.append({'candidate':name,'macro_f1_mean':float(a.mean()),'macro_f1_std':float(a.std(ddof=1)),
            'dropout_brier_mean':float(np.mean([r['dropout_brier'] for r in rows])),
            'dropout_ap_mean':float(np.mean([r['dropout_average_precision'] for r in rows]))})
    # Ties preserve predefined candidate order. No holdout or policy labels used here.
    winner=max([r for r in candidate_summaries if r['candidate']!='dummy'],key=lambda r:r['macro_f1_mean'])['candidate']
    print('Selected before policy/holdout evaluation:',winner,flush=True)
    # Fixed admission-only logistic reference: not another selection/tuning opportunity.
    admission_rows,_=cross_validate_candidate('logistic',ADMISSION_FEATURES,data,train)
    folds.extend(admission_rows)
    pd.DataFrame(folds).to_csv(REPORT_DIR/'cv_folds.csv',index=False)
    pd.DataFrame(candidate_summaries).to_csv(REPORT_DIR/'model_comparison.csv',index=False)
    model=candidate_model(winner,FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(data.iloc[train][FEATURES],data.iloc[train].Status)
    policy_p=align_probabilities(model,data.iloc[policy][FEATURES])
    threshold_rows=[policy_metrics(data.iloc[policy].Status,policy_p[:,0],t) for t in np.round(np.arange(.05,.951,.01),2)]
    best=max(threshold_rows,key=lambda r:(r['f2'],r['threshold']))
    threshold=best['threshold']
    pd.DataFrame(threshold_rows).to_csv(REPORT_DIR/'threshold_analysis.csv',index=False)
    # Freeze the model and policy BEFORE examining the historical holdout.
    model_path=ARTIFACT_DIR/'model.joblib';joblib.dump(model,model_path,compress=3)
    environment={p:version(p) for p in ['scikit-learn','numpy','pandas','scipy','joblib','matplotlib']}
    environment['python']=platform.python_version()
    manifest={'model_version':'semester1-v1.0.0','candidate':winner,'classes':CLASS_NAMES,
        'features':FEATURES,'threshold':threshold,'threshold_objective':'F2 on policy validation',
        'data_sha256':sha256_file(DATA_PATH),'model_sha256':sha256_file(model_path),
        'protocol':PROTOCOL,'environment':environment,
        'training_ranges':{c:{'min':float(data.iloc[train][c].min()),'max':float(data.iloc[train][c].max())} for c in NUMERICAL},
        'counts':{'model_development':len(train),'policy_validation':len(policy),'historical_holdout':len(holdout)},
        'excluded_features':[c for c in data if c not in FEATURES and c!='Status'],
        'code_sha256':{p.name:sha256_file(p) for p in sorted((ROOT/'student_success').glob('*.py'))}}
    write_json(ARTIFACT_DIR/'manifest.json',manifest)
    write_json(ARTIFACT_DIR/'feature_schema.json',schema_records())
    holdout_p=align_probabilities(model,data.iloc[holdout][FEATURES])
    dummy=DummyClassifier(strategy='prior').fit(data.iloc[train][FEATURES],data.iloc[train].Status)
    dummy_p=align_probabilities(dummy,data.iloc[holdout][FEATURES])
    metrics={'protocol':PROTOCOL,'selected_candidate':winner,'threshold':threshold,'counts':manifest['counts'],
        'model_selection':candidate_summaries,
        'admission_logistic_cv_macro_f1':float(np.mean([r['macro_f1'] for r in admission_rows])),
        'policy_validation':evaluate(data.iloc[policy].Status,policy_p,threshold),
        'historical_holdout':evaluate(data.iloc[holdout].Status,holdout_p,threshold),
        'dummy_historical_holdout':evaluate(data.iloc[holdout].Status,dummy_p),
        'bootstrap_95ci':bootstrap_intervals(data.iloc[holdout].Status,holdout_p,threshold)}
    write_json(REPORT_DIR/'metrics.json',metrics)
    for label,idx,probs in [('historical_holdout',holdout,holdout_p),('policy_validation',policy,policy_p)]:
        frame=pd.DataFrame({'source_row':idx+1,'actual':data.iloc[idx].Status.to_numpy(),
            'predicted':np.asarray(CLASS_NAMES)[probs.argmax(axis=1)]})
        for j,c in enumerate(CLASS_NAMES):frame['prob_'+c.lower()]=probs[:,j]
        frame['review']=probs[:,0]>=threshold
        frame.to_csv(REPORT_DIR/f'{label}_predictions.csv',index=False)
    # Post-freeze descriptive diagnostics; never feed these back into selection.
    sub=data.iloc[holdout];group_rows=[]
    groups={'Gender':sub.Gender.astype(str),'Age_group':np.where(sub.Age_at_enrollment<=24,'16–24','25+'),
        'Scholarship_holder':sub.Scholarship_holder.astype(str)}
    for attribute,values in groups.items():
        values=np.asarray(values)
        for value in sorted(set(values)):
            mask=values==value;p=policy_metrics(sub.Status.to_numpy()[mask],holdout_p[mask,0],threshold)
            group_rows.append({'attribute':attribute,'group':value,'n':int(mask.sum()),
                'actual_dropout':int((sub.Status.to_numpy()[mask]=='Dropout').sum()),**p})
    pd.DataFrame(group_rows).to_csv(REPORT_DIR/'subgroup_metrics.csv',index=False)
    importance=permutation_importance(model,sub[FEATURES],sub.Status,scoring='f1_macro',n_repeats=8,random_state=SEED,n_jobs=1)
    pd.DataFrame({'feature':FEATURES,'mean_macro_f1_decrease':importance.importances_mean,
        'std':importance.importances_std}).sort_values('mean_macro_f1_decrease',ascending=False).to_csv(REPORT_DIR/'permutation_importance.csv',index=False)
    quality={'rows':len(data),'columns':len(data.columns),'explicit_nulls':int(data.isna().sum().sum()),
        'duplicate_rows':int(data.duplicated().sum()),'status_counts':{k:int(v) for k,v in data.Status.value_counts().items()},
        'unknown_parent_qualification':{c:int(data[c].eq(34).sum()) for c in ['Mothers_qualification','Fathers_qualification']},
        'selected_schema_valid':True,'data_sha256':sha256_file(DATA_PATH)}
    write_json(REPORT_DIR/'data_quality.json',quality)
    examples=pd.DataFrame([defaults(n) for n in ['Contoh umum','Perlu dukungan akademik','Akademik kuat']])
    examples.to_csv(ROOT/'examples'/'students_template.csv',index=False)
    from .visuals import build_figures
    build_figures()
    print(json.dumps({'winner':winner,'threshold':threshold,'holdout':metrics['historical_holdout']},indent=2),flush=True)

if __name__=='__main__':main()
