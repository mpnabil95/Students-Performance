"""Figures are generated from recorded results, never manually entered metrics."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from .config import DATA_PATH,REPORT_DIR,CLASS_NAMES

COLORS={'Dropout':'#d46b41','Enrolled':'#b19042','Graduate':'#19857b'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':17,
        'axes.titleweight':'bold','axes.titlepad':18,'axes.labelcolor':'#344054',
        'text.color':'#142b3b','xtick.color':'#475467','ytick.color':'#475467',
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'axes.edgecolor':'#d0d5dd','figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','axes.grid':False})

def save(fig,name):
    folder=REPORT_DIR/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'{name}.png',dpi=150,bbox_inches='tight')
    fig.savefig(folder/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def build_figures():
    style();data=pd.read_csv(DATA_PATH,sep=';')
    metrics=json.loads((REPORT_DIR/'metrics.json').read_text())
    pred=pd.read_csv(REPORT_DIR/'historical_holdout_predictions.csv')
    counts=data.Status.value_counts().reindex(CLASS_NAMES)
    fig,ax=plt.subplots(figsize=(9,4.7));bars=ax.barh(CLASS_NAMES,counts,color=[COLORS[c] for c in CLASS_NAMES],height=.55)
    for bar,c in zip(bars,counts):ax.text(c+35,bar.get_y()+bar.get_height()/2,f'{c:,}  ·  {c/len(data):.1%}',va='center')
    ax.set_xlim(0,max(counts)*1.3);ax.set_title('Status studi pada dataset historis',loc='left');ax.set_xlabel('Jumlah mahasiswa (n = 4.424)')
    save(fig,'status_distribution')
    cv=pd.read_csv(REPORT_DIR/'model_comparison.csv')
    fig,ax=plt.subplots(figsize=(9,4.8));ax.barh(cv.candidate,cv.macro_f1_mean,xerr=cv.macro_f1_std,color=['#c3cbd2','#75a9c0','#458499','#19857b'],height=.55,capsize=4)
    ax.set_xlim(0,1);ax.set_title('Pemilihan model pada data pengembangan',loc='left');ax.set_xlabel('Macro F1 · rerata ± simpangan baku 5 fold')
    fig.tight_layout();save(fig,'model_selection')
    cm=np.array(metrics['historical_holdout']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(7,5.5));ax.imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,color='white' if cm[i,j]>cm.max()*.5 else '#142b3b')
    ax.set_xticks(range(3),CLASS_NAMES);ax.set_yticks(range(3),CLASS_NAMES);ax.set_xlabel('Prediksi');ax.set_ylabel('Aktual');ax.set_title('Confusion matrix · holdout historis',loc='left')
    save(fig,'confusion_matrix')
    actual=pred.actual.eq('Dropout').to_numpy();p=pred.prob_dropout.to_numpy()
    precision,recall,_=precision_recall_curve(actual,p)
    point=metrics['historical_holdout']['policy']
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(recall,precision,color='#19857b',lw=2.5,label='Model terkalibrasi')
    ax.axhline(actual.mean(),color='#8896a2',ls='--',label=f'Prevalensi Dropout = {actual.mean():.1%}')
    ax.scatter([point['recall']],[point['precision']],s=100,color='#d46b41',zorder=5,label=f"Threshold validasi = {metrics['threshold']:.2f}")
    ax.set(xlim=(0,1.02),ylim=(0,1.02),xlabel='Recall Dropout',ylabel='Precision Dropout');ax.set_title('Trade-off peninjauan mahasiswa',loc='left');ax.legend(frameon=False,loc='lower left')
    save(fig,'precision_recall')
    true,mean=calibration_curve(actual,p,n_bins=8,strategy='quantile')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));axes[0].plot([0,1],[0,1],ls='--',color='#8896a2',label='Kalibrasi ideal');axes[0].plot(mean,true,'o-',color='#19857b',label='Holdout historis')
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Rata-rata probabilitas',ylabel='Proporsi Dropout aktual');axes[0].legend(frameon=False);axes[0].set_title('Reliability curve',loc='left')
    axes[1].hist(p,bins=np.linspace(0,1,11),color='#458499',edgecolor='white');axes[1].set(xlabel='Probabilitas Dropout',ylabel='Jumlah mahasiswa');axes[1].set_title('Sebaran probabilitas',loc='left')
    fig.tight_layout();save(fig,'calibration')
    th=pd.read_csv(REPORT_DIR/'threshold_analysis.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    for c,color in [('recall','#19857b'),('precision','#458499'),('review_rate','#b19042'),('f2','#d46b41')]:ax.plot(th.threshold,th[c],label=c,color=color,lw=2)
    ax.axvline(metrics['threshold'],color='#142b3b',ls='--',alpha=.7);ax.set(xlabel='Threshold probabilitas Dropout',ylabel='Nilai',ylim=(0,1.03));ax.legend(ncol=4,frameon=False,loc='lower center');ax.set_title('Kebijakan dipilih pada policy validation',loc='left')
    save(fig,'threshold_selection')
    imp=pd.read_csv(REPORT_DIR/'permutation_importance.csv').head(10).iloc[::-1]
    from .schema import FIELDS
    fig,ax=plt.subplots(figsize=(10,5.5));ax.barh([FIELDS[f].label for f in imp.feature],imp.mean_macro_f1_decrease,xerr=imp['std'],color='#19857b',capsize=3,height=.6)
    ax.axvline(0,color='#8896a2',lw=1);ax.set_xlabel('Penurunan macro F1 setelah fitur diacak · mean ± SD');ax.set_title('Permutation importance · holdout historis',loc='left');fig.tight_layout();save(fig,'feature_importance')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,col,title in zip(axes,['Curricular_units_1st_sem_approved','Curricular_units_1st_sem_grade'],['Unit lulus semester 1','Nilai semester 1']):
        values=[data.loc[data.Status==s,col] for s in CLASS_NAMES]
        bp=ax.boxplot(values,tick_labels=CLASS_NAMES,patch_artist=True,showfliers=False)
        for patch,c in zip(bp['boxes'],CLASS_NAMES):patch.set_facecolor(COLORS[c]);patch.set_alpha(.7)
        ax.set_title(title,loc='left');ax.set_ylabel('Nilai historis');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save(fig,'academic_patterns')

if __name__=='__main__':build_figures()
