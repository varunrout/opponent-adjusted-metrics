from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from google.cloud import bigquery, storage
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px

PROJECT='oam-varun-260819'; DATASET='oam_analysis'; RUN='cxg-analysis-20260820T201934Z'; BUCKET='oam-varun-260819-artifacts'; PREFIX='analysis/cxg'
OUT=Path('audit_outputs/cxg_analysis')/RUN/'baseline_multivariate_charts'
OUT.mkdir(parents=True, exist_ok=True)
bq=bigquery.Client(project=PROJECT); gcs=storage.Client(project=PROJECT)
rows=[dict(r.items()) for r in bq.query(f"SELECT model_name,model_stage,cohort,families,feature_count,row_count,goal_count,auc,log_loss,brier FROM `{PROJECT}.{DATASET}.cxg_baseline_multivariate_results_v1` ORDER BY auc DESC", location='europe-west2').result()]
df=pd.DataFrame(rows)
charts=[]

def write_bar(metric,title,name):
    tmp=df.sort_values(metric, ascending=True)
    fig=px.bar(tmp, x=metric, y='model_name', color='cohort', orientation='h', hover_data=['model_stage','families','feature_count','row_count','goal_count'], title=title)
    fig.update_layout(template='plotly_white', font={'family':'Arial','size':13}, margin={'l':80,'r':30,'t':72,'b':72})
    html=OUT/f'{name}.html'; png=OUT/f'{name}.png'
    fig.write_html(html, include_plotlyjs='cdn', full_html=True)
    plt_fig,ax=plt.subplots(figsize=(14.5,9.2), dpi=150)
    colors={'event_all':'#2563eb','360_cohort':'#dc2626'}
    ax.barh(range(len(tmp)), tmp[metric], color=[colors.get(v,'#64748b') for v in tmp['cohort']], alpha=.86)
    ax.set_yticks(range(len(tmp))); ax.set_yticklabels([str(v)[:58] for v in tmp['model_name']], fontsize=7)
    ax.set_xlabel(metric); ax.grid(axis='x', alpha=.18); ax.set_title(title, fontsize=14, pad=14)
    handles=[plt.Rectangle((0,0),1,1,color=c) for c in colors.values()]
    ax.legend(handles, list(colors.keys()), fontsize=8, loc='best')
    plt_fig.tight_layout(); plt_fig.savefig(png); plt.close(plt_fig)
    return html,png
for metric,title,name in [('auc','Baseline/candidate/experiment ranking by AUC','baseline_auc_ranking'),('log_loss','Baseline/candidate/experiment ranking by log loss','baseline_log_loss_ranking')]:
    html,png=write_bar(metric,title,name)
    for path in [html,png]:
        ct='text/html' if path.suffix=='.html' else 'image/png'
        gcs.bucket(BUCKET).blob(f'{PREFIX}/{RUN}/baseline_multivariate_charts/{path.name}').upload_from_filename(path, content_type=ct)
    charts.append({'run_id':RUN,'chart_name':name,'chart_title':title,'html_uri':f'gs://{BUCKET}/{PREFIX}/{RUN}/baseline_multivariate_charts/{html.name}','png_uri':f'gs://{BUCKET}/{PREFIX}/{RUN}/baseline_multivariate_charts/{png.name}'})
manifest={'run_id':RUN,'rendered_at':datetime.now(timezone.utc).isoformat(),'chart_count':len(charts),'charts':charts,'local_prefix':str(OUT),'artifact_prefix':f'gs://{BUCKET}/{PREFIX}/{RUN}/baseline_multivariate_charts/'}
mp=OUT/'baseline_multivariate_charts_manifest.json'; mp.write_text(json.dumps(manifest,indent=2), encoding='utf-8')
gcs.bucket(BUCKET).blob(f'{PREFIX}/{RUN}/baseline_multivariate_charts/{mp.name}').upload_from_filename(mp, content_type='application/json')
selects=[]
for c in charts:
    selects.append("SELECT '{run_id}' AS run_id, '{chart_name}' AS chart_name, '{chart_title}' AS chart_title, '{html_uri}' AS html_uri, '{png_uri}' AS png_uri, CURRENT_TIMESTAMP() AS rendered_at".format(**c))
bq.query(f"CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.cxg_baseline_chart_registry_v1` AS\n"+'\nUNION ALL\n'.join(selects), location='europe-west2').result()
print(json.dumps(manifest, indent=2))

