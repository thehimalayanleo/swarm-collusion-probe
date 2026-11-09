import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
E=[json.loads(l) for l in open("episodes.jsonl")]; X=np.load("feats.npy")[:,1].astype(np.float32)
m=np.array([e["arm"] in "CLB" for e in E]); y=np.array([e["arm"]=="C" for e in E]); fam=np.array([str(e["family"]) for e in E])
R=[e["reply"] for e in E]
def lodo(feat):
    out=[]
    for f in sorted(set(fam)):
        tr=m&(fam!=f); te=m&(fam==f); out.append(roc_auc_score(y[te],feat(tr,te)))
    return round(float(np.mean(out)),3), round(float(min(out)),3)
def probe(tr,te):
    sc=StandardScaler().fit(X[tr]); c=LogisticRegression(C=0.01,max_iter=3000).fit(sc.transform(X[tr]),y[tr]); return c.decision_function(sc.transform(X[te]))
def tf(tr,te):
    a=[R[i] for i in np.where(tr)[0]]; b=[R[i] for i in np.where(te)[0]]
    v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(a)
    c=LogisticRegression(C=10,max_iter=3000).fit(v.transform(a),y[tr]); return c.decision_function(v.transform(b))
print("probe L18 C vs L+B, leave-one-domain-out AUROC mean/min", lodo(probe))
print("tfidf reply C vs L+B, leave-one-domain-out AUROC mean/min", lodo(tf))
