"""Models and metrics; preprocessing is always fit inside the CV fold."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
    log_loss, brier_score_loss, average_precision_score, precision_score, recall_score, fbeta_score)
from .config import SEED, CLASS_NAMES
from .schema import CATEGORICAL

def candidate_model(name, features):
    cats=[c for c in features if c in CATEGORICAL]
    nums=[c for c in features if c not in cats]
    prep=ColumnTransformer([
        ('numeric',StandardScaler(),nums),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),
    ],remainder='drop')
    estimators={
        'logistic': LogisticRegression(C=1.0,max_iter=4000,solver='lbfgs'),
        'random_forest': RandomForestClassifier(n_estimators=180,max_depth=12,min_samples_leaf=4,
            class_weight='balanced_subsample',random_state=SEED,n_jobs=1),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,
            learning_rate=.08,l2_regularization=1.0,early_stopping=False,random_state=SEED),
    }
    pipeline=Pipeline([('preprocess',prep),('classifier',estimators[name])])
    return CalibratedClassifierCV(pipeline,method='sigmoid',ensemble=True,
        cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED),n_jobs=1)

def align_probabilities(model, x):
    probs=model.predict_proba(x)
    return probs[:,[list(model.classes_).index(c) for c in CLASS_NAMES]]

def policy_metrics(y, dropout_p, threshold):
    actual=np.asarray(y)=='Dropout'; flag=np.asarray(dropout_p)>=threshold
    tp=int(np.sum(actual & flag));fn=int(np.sum(actual & ~flag))
    fp=int(np.sum(~actual & flag));tn=int(np.sum(~actual & ~flag))
    return {'threshold':float(threshold),'precision':float(precision_score(actual,flag,zero_division=0)),
        'recall':float(recall_score(actual,flag,zero_division=0)),
        'f2':float(fbeta_score(actual,flag,beta=2,zero_division=0)),
        'review_rate':float(flag.mean()),'review_count':int(flag.sum()),
        'tp':tp,'fn':fn,'fp':fp,'tn':tn}

def evaluate(y, probs, threshold=None):
    y=np.asarray(y);pred=np.asarray(CLASS_NAMES)[np.argmax(probs,axis=1)]
    result={'n':len(y),'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')),
        'weighted_f1':float(f1_score(y,pred,average='weighted')),
        'log_loss':float(log_loss(y,probs,labels=CLASS_NAMES)),
        'dropout_brier':float(brier_score_loss(y=='Dropout',probs[:,0])),
        'dropout_average_precision':float(average_precision_score(y=='Dropout',probs[:,0])),
        'classification_report':classification_report(y,pred,labels=CLASS_NAMES,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y,pred,labels=CLASS_NAMES).tolist()}
    if threshold is not None:result['policy']=policy_metrics(y,probs[:,0],threshold)
    return result

def bootstrap_intervals(y,probs,threshold,repeats=500):
    """Fixed-model, row bootstrap; does not include training/selection uncertainty."""
    rng=np.random.default_rng(SEED);y=np.asarray(y);store={'macro_f1':[],'dropout_recall_policy':[],'dropout_precision_policy':[]}
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y));a=y[idx];p=probs[idx]
        pred=np.asarray(CLASS_NAMES)[p.argmax(axis=1)]
        store['macro_f1'].append(f1_score(a,pred,labels=CLASS_NAMES,average='macro',zero_division=0))
        store['dropout_recall_policy'].append(recall_score(a=='Dropout',p[:,0]>=threshold,zero_division=0))
        store['dropout_precision_policy'].append(precision_score(a=='Dropout',p[:,0]>=threshold,zero_division=0))
    return {k:{'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975))} for k,v in store.items()}
