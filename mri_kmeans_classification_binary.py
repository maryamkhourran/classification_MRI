"""
╔═══════════════════════════════════════════════════════════════════╗
║   Classification MRI Cérébrale — SYSTÈME COMPLET                ║
║   VERSION DÉFINITIVE — CAD System (Computer-Aided Diagnosis)    ║
║                                                                   ║
║   PIPELINE :                                                      ║
║     Prétraitement → Segmentation → 13 Features → Modèles        ║
║                                                                   ║
║   MODÈLES COMPARÉS :                                             ║
║     ① K-Means (non supervisé, from scratch)                     ║
║     ② Random Forest (supervisé)                                  ║
║     ③ SVM (supervisé)                                            ║
║     ④ Logistic Regression (supervisé)                            ║
║                                                                   ║
║   SORTIES :                                                       ║
║     ✅ Train/Test Split stratifié (80/20)                        ║
║     ✅ Matrice de confusion par modèle                           ║
║     ✅ Courbe ROC + AUC pour chaque modèle                       ║
║     ✅ Feature importance (Random Forest)                         ║
║     ✅ Comparaison graphique des 4 modèles                       ║
╚═══════════════════════════════════════════════════════════════════╝
"""
# This script powers the binary K-Means workflow shown in the project.
# It reuses the same source MRI dataset as the multiclass pipeline, then
# groups the labels into two targets: tumor versus no_tumor.


import os, glob, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
IMG_SIZE     = 128
N_CLUSTERS   = 2
N_SAMPLES    = 50
KMEANS_ITER  = 150
KMEANS_NINIT = 10
TEST_RATIO   = 0.20
SEED         = 42
CLASSES      = ["tumor", "no_tumor"]
COLORS       = ["#E74C3C", "#3498DB"]
MODEL_COLORS = {"K-Means": "#F39C12", "Random Forest": "#2ECC71",
                "SVM": "#9B59B6", "Logistic Reg.": "#00D4FF"}

np.random.seed(SEED)

FEATURE_NAMES = [
    "Intensité", "Std", "Taille ROI",
    "Contraste", "Homogénéité", "Énergie", "Corrélation",
    "Compacité", "Skewness", "Kurtosis",
    "LBP Entropie", "Gradient moy.", "Entropie locale"
]


# ═══════════════════════════════════════════════════════════════════
#  PRÉ-TRAITEMENT
# ═══════════════════════════════════════════════════════════════════

def gaussian_kernel(size=5, sigma=1.4):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return g / g.sum()

def conv2d(img, kernel):
    try:
        from scipy.signal import convolve2d
        return convolve2d(img.astype(float), kernel, mode='same', boundary='symm')
    except ImportError:
        kh, kw = kernel.shape
        ph, pw = kh//2, kw//2
        padded = np.pad(img.astype(float), ((ph,ph),(pw,pw)), mode="reflect")
        out = np.zeros_like(img, dtype=float)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i,j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
        return out

def equalize_hist(img):
    hist, _ = np.histogram(img.flatten(), 256, (0,256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf>0].min()
    lut = np.round((cdf - cdf_min)/(img.size - cdf_min)*255).astype(np.uint8)
    return lut[img]

def normalize(img):
    mn, mx = img.min(), img.max()
    return (img.astype(float) - mn) / (mx - mn + 1e-8)

GK = gaussian_kernel(5, 1.4)

def preprocess(img_uint8):
    blurred   = conv2d(img_uint8, GK)
    equalized = equalize_hist(np.clip(blurred,0,255).astype(np.uint8))
    return normalize(equalized)


# ═══════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ═══════════════════════════════════════════════════════════════════

def blur_score(img):
    return np.var(conv2d(img, np.array([[0,1,0],[1,-4,1],[0,1,0]])))

def select_best_images(files, n=50):
    scored = []
    for idx, f in enumerate(files):
        if (idx+1) % 10 == 0 or idx == 0:
            print(f"     Analyse... {idx+1}/{len(files)}", end='\r', flush=True)
        try:
            img = np.array(Image.open(f).convert("L").resize((IMG_SIZE,IMG_SIZE)))
            scored.append((blur_score(img)+img.std()-abs(img.mean()-128), f))
        except: continue
    print(" "*60, end='\r', flush=True)
    scored.sort(key=lambda x: x[0], reverse=True)
    sel = [f for _,f in scored if np.random.rand()>0.2][:n]
    if len(sel) < n:
        sel.extend([f for _,f in scored if f not in sel][:n-len(sel)])
    return sel[:n]

def load_images(dataset_path):
    print("\n" + "═"*65)
    print("  ÉTAPE 1 — Chargement des images MRI")
    print("═"*65)
    images_raw, labels = [], []
    for cls in ["glioma_tumor","meningioma_tumor","pituitary_tumor","no_tumor"]:
        path  = os.path.join(dataset_path, cls)
        files = (glob.glob(os.path.join(path,"*.jpg")) +
                 glob.glob(os.path.join(path,"*.jpeg")) +
                 glob.glob(os.path.join(path,"*.png")))
        files = select_best_images(files, N_SAMPLES)
        for f in files:
            img = Image.open(f).convert("L").resize((IMG_SIZE,IMG_SIZE))
            images_raw.append(np.array(img, dtype=np.uint8))
            labels.append(1 if cls=="no_tumor" else 0)
        print(f"  ✔  {cls:<22} → {len(files)} images")
    return np.array(images_raw), np.array(labels)

def preprocess_all(images_raw, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 2 — Pré-traitement")
    print("═"*65)
    processed = [preprocess(img) for img in images_raw]
    processed = np.array(processed)

    unique_labels = np.unique(labels)
    fig, axes = plt.subplots(len(unique_labels), 4, figsize=(15,14))
    if len(unique_labels)==1: axes=axes.reshape(1,-1)
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 2 — Pipeline de pré-traitement",fontsize=14,fontweight="bold",color="white")
    for c,t in enumerate(["Brute","Flou Gaussien","Égalisation","Normalisée"]):
        axes[0][c].set_title(t,color="white",fontsize=10,fontweight="bold",pad=6)
    for r,ci in enumerate(unique_labels):
        idx = np.where(labels==ci)[0][0]
        raw = images_raw[idx]
        bl  = np.clip(conv2d(raw,GK),0,255).astype(np.uint8)
        eq  = equalize_hist(bl)
        nm  = normalize(eq)
        for c,(step,cm) in enumerate(zip([raw,bl,eq,nm],["gray","gray","gray","inferno"])):
            axes[r][c].imshow(step,cmap=cm); axes[r][c].axis("off")
            if c==0: axes[r][c].set_ylabel("Tumeur" if ci==0 else "Pas tumeur",
                                            color=COLORS[ci],fontsize=10,fontweight="bold")
    plt.tight_layout()
    plt.savefig("etape2_pretraitement.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape2_pretraitement.png")
    return processed




from skimage import segmentation, morphology, exposure, measure, filters

def segment_tumor_chanvese(img):
    # Normalisation
    img = exposure.rescale_intensity(img)

    # Débruitage léger
    img_denoised = filters.median(img, morphology.disk(3))

    # Chan-Vese
    mask = segmentation.chan_vese(
        img_denoised,
        mu=0.2,
        lambda1=1,
        lambda2=1,
        tol=1e-3,
        max_num_iter=200
    )

    # Nettoyage
    mask = morphology.remove_small_objects(mask, min_size=300)
    mask = morphology.binary_closing(mask, morphology.disk(3))

    # Plus grande région (tumeur)
    labels = measure.label(mask)
    if labels.max() > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == counts.argmax()

    return mask.astype(bool)




# ═══════════════════════════════════════════════════════════════════
#  SEGMENTATION AVANCÉE (visualisation uniquement)
# ═══════════════════════════════════════════════════════════════════

def otsu_threshold(img_norm):
    img8 = (img_norm*255).astype(np.uint8)
    hist,_ = np.histogram(img8.flatten(),256,(0,256))
    total=img8.size; best_t,best_var=0,-1; wb,sb=0,0
    S=sum(i*hist[i] for i in range(256))
    for t in range(256):
        wb+=hist[t]; wf=total-wb
        if wb==0 or wf==0: continue
        sb+=t*hist[t]; mb=sb/wb; mf=(S-sb)/wf
        var=wb*wf*(mb-mf)**2
        if var>best_var: best_var,best_t=var,t
    return best_t/255.0

def morpho_op(mask,iters,dilate=False):
    r=mask.copy()
    for _ in range(iters):
        h,w=r.shape; pad=np.pad(r,1,mode="constant")
        new=(np.zeros if dilate else np.ones)((h,w),dtype=bool)
        for di in range(3):
            for dj in range(3):
                if dilate: new|=pad[di:di+h,dj:dj+w]
                else:      new&=pad[di:di+h,dj:dj+w]
        r=new
    return r

def largest_comp(mask):
    h,w=mask.shape; lab=np.zeros((h,w),dtype=int); cur=0
    for i in range(h):
        for j in range(w):
            if mask[i,j] and lab[i,j]==0:
                cur+=1; q=[(i,j)]; lab[i,j]=cur
                while q:
                    ci,cj=q.pop(0)
                    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni,nj=ci+di,cj+dj
                        if 0<=ni<h and 0<=nj<w and mask[ni,nj] and lab[ni,nj]==0:
                            lab[ni,nj]=cur; q.append((ni,nj))
    if cur==0: return np.zeros_like(mask)
    best=np.argmax([(lab==k).sum() for k in range(1,cur+1)])+1
    return (lab==best).astype(bool)

def get_brain_mask(img_norm):
    h,w=img_norm.shape; tissue=img_norm>0.10
    visited=np.zeros((h,w),dtype=bool); q=[]
    for i in range(h):
        for j in [0,w-1]:
            if not tissue[i,j] and not visited[i,j]: visited[i,j]=True; q.append((i,j))
    for j in range(w):
        for i in [0,h-1]:
            if not tissue[i,j] and not visited[i,j]: visited[i,j]=True; q.append((i,j))
    while q:
        ci,cj=q.pop()
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni,nj=ci+di,cj+dj
            if 0<=ni<h and 0<=nj<w and not tissue[ni,nj] and not visited[ni,nj]:
                visited[ni,nj]=True; q.append((ni,nj))
    head=tissue&~visited
    brain=morpho_op(head,6)
    if brain.sum()<200: brain=morpho_op(head,3)
    return brain



def overlay_tumor(img_norm,mask,color=(1,0,0)):
    rgb=np.stack([img_norm]*3,axis=-1)
    rgb[mask,0]=color[0]; rgb[mask,1]=color[1]; rgb[mask,2]=color[2]
    return np.clip(rgb,0,1)

def visualize_segmentation(images_proc, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 3 — Segmentation (visualisation)")
    print("═"*65)

    for idx, suf in [
        (np.where(labels==0)[0][min(2, len(np.where(labels==0)[0])-1)], "tumor"),
        (np.where(labels==1)[0][min(2, len(np.where(labels==1)[0])-1)], "sain")
    ]:
        img = images_proc[idx]
        mask = segment_tumor_chanvese(img)

        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        fig.patch.set_facecolor("#0D0D0D")

        fig.suptitle(f"ÉTAPE 3 — {suf}", color="white", fontweight="bold")

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Image filtrée", color="white")
        axes[0].axis("off")

        axes[1].imshow(overlay_tumor(img, mask))
        axes[1].set_title("Segmentation Chan-Vese", color="white")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"etape3_{suf}.png", dpi=130)
        plt.close()


# ═══════════════════════════════════════════════════════════════════
#  EXTRACTION DES 13 FEATURES
# ═══════════════════════════════════════════════════════════════════

def compute_lbp(img_norm):
    h,w=img_norm.shape; lbp=np.zeros((h,w),dtype=np.uint8)
    img8=(img_norm*255).astype(np.uint8)
    neighbors=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i in range(1,h-1):
        for j in range(1,w-1):
            center=img8[i,j]; code=0
            for bit,(di,dj) in enumerate(neighbors):
                if img8[i+di,j+dj]>=center: code|=(1<<bit)
            lbp[i,j]=code
    hist,_=np.histogram(lbp.flatten(),bins=256,range=(0,256))
    hist=hist.astype(float)/(hist.sum()+1e-8)
    bins=np.arange(256)
    lbp_mean=np.sum(bins*hist); lbp_var=np.sum((bins-lbp_mean)**2*hist)
    lbp_energy=np.sum(hist**2)
    lbp_entropy=-np.sum(hist[hist>0]*np.log2(hist[hist>0]+1e-8))
    return lbp_mean,np.sqrt(lbp_var),lbp_energy,lbp_entropy

def compute_gradient(img_norm):
    gx=conv2d(img_norm,np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=float))
    gy=conv2d(img_norm,np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=float))
    mag=np.sqrt(gx**2+gy**2)
    return float(mag.mean()),float(mag.std()),float(mag.max())

def compute_local_entropy(img_norm,mask):
    roi=img_norm[mask]
    if len(roi)<4: return 0.0
    hist,_=np.histogram((roi*255).astype(np.uint8),bins=256,range=(0,256))
    h=hist.astype(float)/(hist.sum()+1e-8)
    return float(-np.sum(h[h>0]*np.log2(h[h>0]+1e-8)))

def extract_features(images,labels):
    from skimage.feature import graycomatrix,graycoprops
    from skimage.measure import perimeter
    all_features,valid_indices=[],[]
    eps=1e-8
    for img_idx,img in enumerate(images):
        if (img_idx+1)%50==0:
            print(f"  ... features {img_idx+1}/{len(images)}",end='\r',flush=True)
        mask = segment_tumor_chanvese(img); mask_sum=np.sum(mask)
        if mask_sum<16: continue
        roi=img[mask]; p1,p99=np.percentile(roi,(1,99))
        if p99-p1<eps: continue
        roi_c=np.clip(roi,p1,p99)
        coords=np.argwhere(mask); y0,x0=coords.min(axis=0); y1,x1=coords.max(axis=0)+1
        roi_crop=np.clip(img[y0:y1,x0:x1],p1,p99)
        roi_32=((roi_crop-p1)/(p99-p1+eps)*31).astype(np.uint8)
        glcm=graycomatrix(roi_32,distances=[1],angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                          levels=32,symmetric=True,normed=True)
        mean_roi=np.mean(roi_c); std_roi=np.std(roi_c)+eps; diff=roi_c-mean_roi
        perim=perimeter(mask); compact=(4*np.pi*mask_sum)/(perim**2+eps)
        _,_,_,lbp_ent=compute_lbp(img)
        grad_mean,_,_=compute_gradient(img)
        loc_ent=compute_local_entropy(img,mask)
        feat=[mean_roi,std_roi,mask_sum/mask.size,
              np.mean(graycoprops(glcm,'contrast')),np.mean(graycoprops(glcm,'homogeneity')),
              np.mean(graycoprops(glcm,'energy')),np.mean(graycoprops(glcm,'correlation')),
              compact,np.mean(diff**3)/(std_roi**3),np.mean(diff**4)/(std_roi**4),
              lbp_ent,grad_mean,loc_ent]
        all_features.append(feat); valid_indices.append(img_idx)
    print(" "*60,end='\r',flush=True)
    return np.array(all_features),np.array(valid_indices)

def zscore(X):
    mu=X.mean(axis=0); sig=X.std(axis=0); sig[sig==0]=1
    return (X-mu)/sig,mu,sig

def pca_2d(X):
    Xc=X-X.mean(axis=0); cov=Xc.T@Xc/len(Xc)
    vals,vecs=np.linalg.eigh(cov); idx=np.argsort(vals)[::-1]
    vecs=vecs[:,idx]; expl=vals[idx]/vals.sum()*100
    return Xc@vecs[:,:2],vecs[:,:2],expl[:2]


# ═══════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT STRATIFIÉ
# ═══════════════════════════════════════════════════════════════════

def train_test_split_stratified(X,y,test_ratio=TEST_RATIO,seed=SEED):
    rng=np.random.RandomState(seed); train_idx,test_idx=[],[]
    for cls in np.unique(y):
        ci=np.where(y==cls)[0]; rng.shuffle(ci)
        n_te=max(1,int(len(ci)*test_ratio))
        test_idx.extend(ci[:n_te].tolist()); train_idx.extend(ci[n_te:].tolist())
    train_idx=np.array(train_idx); test_idx=np.array(test_idx)
    print(f"\n  Split : train={len(train_idx)}  test={len(test_idx)}")
    for cls in np.unique(y):
        print(f"  {CLASSES[cls]:<12} → train={(y[train_idx]==cls).sum()}  test={(y[test_idx]==cls).sum()}")
    return train_idx,test_idx


# ═══════════════════════════════════════════════════════════════════
#  ① K-MEANS FROM SCRATCH (multi-init)
# ═══════════════════════════════════════════════════════════════════

class KMeans:
    def __init__(self,k=2,max_iter=KMEANS_ITER,tol=1e-5,n_init=KMEANS_NINIT):
        self.k,self.max_iter,self.tol,self.n_init=k,max_iter,tol,n_init
        self.centroids=self.labels_=self.inertia_=None
        self.inertia_history=[]

    def _init_pp(self,X,rng):
        c=[X[rng.randint(len(X))]]
        for _ in range(1,self.k):
            d2=np.array([min(np.sum((x-ci)**2) for ci in c) for x in X])
            c.append(X[np.searchsorted((d2/(d2.sum()+1e-8)).cumsum(),rng.rand())])
        return np.array(c)

    def _assign(self,X,centroids):
        return np.argmin(np.array([[np.sum((x-c)**2) for c in centroids] for x in X]),axis=1)

    def _run_once(self,X,seed):
        rng=np.random.RandomState(seed); centroids=self._init_pp(X,rng); hist=[]
        for _ in range(self.max_iter):
            old=centroids.copy(); lbl=self._assign(X,centroids)
            for k in range(self.k):
                pts=X[lbl==k]
                if len(pts): centroids[k]=pts.mean(axis=0)
            inertia=sum(np.sum((X[lbl==k]-centroids[k])**2) for k in range(self.k))
            hist.append(inertia)
            if np.sqrt(np.sum((centroids-old)**2))<self.tol: break
        lbl=self._assign(X,centroids)
        return centroids,lbl,sum(np.sum((X[lbl==k]-centroids[k])**2) for k in range(self.k)),hist

    def fit(self,X):
        best_inertia=np.inf; best=None
        print(f"  K-Means multi-init (n={self.n_init}) :",end=" ",flush=True)
        for i in range(self.n_init):
            c,l,inertia,hist=self._run_once(X,SEED+i)
            print(f"{i+1}",end=".",flush=True)
            if inertia<best_inertia: best_inertia=inertia; best=(c,l,hist)
        print(f" ✔  inertie={best_inertia:.2f}")
        self.centroids,self.labels_,self.inertia_history=best
        self.inertia_=best_inertia; return self

    def predict(self,X):
        return self._assign(X,self.centroids)

def align_clusters(km_labels,true_labels,k=N_CLUSTERS):
    from itertools import permutations
    best_acc,best_map=0,None
    for perm in permutations(range(k)):
        mapped=np.array([perm[l] for l in km_labels])
        acc=(mapped==true_labels).mean()
        if acc>best_acc: best_acc,best_map=acc,perm
    return best_acc,best_map

def elbow(X_norm):
    print("\n" + "═"*65)
    print("  ÉTAPE 5a — Méthode du coude")
    print("═"*65)
    inertias=[]
    for k in range(2,9):
        km=KMeans(k=k,max_iter=40,n_init=3); km.fit(X_norm)
        inertias.append(km.inertia_); print(f"  k={k}  inertie={km.inertia_:.2f}")
    fig,ax=plt.subplots(figsize=(9,5)); fig.patch.set_facecolor("#1A1A2E"); ax.set_facecolor("#16213E")
    ax.plot(range(2,9),inertias,"o-",color="#00D4FF",lw=2.5,markersize=9,
            markerfacecolor="#E74C3C",markeredgecolor="white",mew=1.5)
    ax.axvline(x=2,color="#F39C12",ls="--",lw=1.8,label="k=2 (binaire)")
    ax.fill_between(range(2,9),inertias,alpha=0.15,color="#00D4FF")
    ax.set_xlabel("Nombre de clusters k",color="white",fontsize=12)
    ax.set_ylabel("Inertie (WCSS)",color="white",fontsize=12)
    ax.set_title("Méthode du coude",color="white",fontsize=13,fontweight="bold")
    ax.tick_params(colors="white"); ax.legend(fontsize=10)
    ax.grid(True,alpha=0.25,ls="--",color="white"); ax.set_xticks(range(2,9))
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("etape5a_coude.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape5a_coude.png")


# ═══════════════════════════════════════════════════════════════════
#  ② ③ ④ MODÈLES SUPERVISÉS (sklearn)
# ═══════════════════════════════════════════════════════════════════

def train_supervised_models(X_train,y_train,X_test,y_test):
    """
    Entraîne et évalue 3 modèles supervisés.
    Retourne un dict avec les résultats de chaque modèle.
    """
    print("\n" + "═"*65)
    print("  ÉTAPE 6b — Modèles supervisés (RF + SVM + LR)")
    print("═"*65)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=4, random_state=SEED, n_jobs=-1),
        "SVM": SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, random_state=SEED),
        "Logistic Reg.": LogisticRegression(
            C=1.0, max_iter=1000, random_state=SEED),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred     = clf.predict(X_test)
        y_prob     = clf.predict_proba(X_test)[:,1]  # prob classe 1 (no_tumor)
        acc        = (y_pred == y_test).mean()
        results[name] = {
            "clf":    clf,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "acc":    acc,
        }
        # Métriques détaillées
        tp=((y_pred==0)&(y_test==0)).sum(); fp=((y_pred==0)&(y_test!=0)).sum()
        fn=((y_pred!=0)&(y_test==0)).sum()
        pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8); f1=2*pr*rc/(pr+rc+1e-8)
        print(f"\n  {name} :")
        print(f"    Accuracy  : {acc*100:.1f}%")
        print(f"    Precision : {pr*100:.1f}%  |  Recall : {rc*100:.1f}%  |  F1 : {f1*100:.1f}%")

    # Feature importance (Random Forest uniquement)
    rf = models["Random Forest"]
    results["__rf_importance__"] = rf.feature_importances_

    return results


# ═══════════════════════════════════════════════════════════════════
#  NUAGES DE POINTS (3 phases)
# ═══════════════════════════════════════════════════════════════════

def visualize_scatter_3phases(km,X_norm,labels,best_map):
    print("\n" + "═"*65)
    print("  ÉTAPE 6 — Nuages de points (3 phases)")
    print("═"*65)
    X2d,vecs,expl=pca_2d(X_norm)
    Xc_mean=X_norm.mean(axis=0); c_2d=(km.centroids-Xc_mean)@vecs
    acc,(bm)=align_clusters(km.labels_,labels)
    mapped=np.array([bm[l] for l in km.labels_]); correct=(mapped==labels)

    fig,axes=plt.subplots(1,3,figsize=(21,7)); fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(f"ÉTAPE 6 — K-Means  |  PCA PC1={expl[0]:.1f}%  PC2={expl[1]:.1f}%",
                 fontsize=14,fontweight="bold",color="white",y=1.02)

    def sp(ax,X2d,cols,title,c2d=None,leg=None):
        ax.set_facecolor("#16213E")
        ax.scatter(X2d[:,0],X2d[:,1],c=cols,s=28,alpha=0.75,edgecolors="none")
        if c2d is not None:
            ax.scatter(c2d[:,0],c2d[:,1],marker="*",s=350,c="white",
                       edgecolors="#2C3E50",linewidths=1.5,zorder=10)
        if leg:
            ax.legend(handles=[mpatches.Patch(color=COLORS[i],label=leg[i]) for i in range(len(leg))],
                      fontsize=8,framealpha=0.6,facecolor="#1A1A2E",labelcolor="white")
        ax.set_title(title,color="white",fontsize=11,fontweight="bold",pad=8)
        ax.set_xlabel("PC1",color="white",fontsize=9); ax.set_ylabel("PC2",color="white",fontsize=9)
        ax.tick_params(colors="white"); ax.grid(True,alpha=0.2,ls="--",color="white")
        for s in ax.spines.values(): s.set_edgecolor("#444")

    sp(axes[0],X2d,["#00D4FF"]*len(X2d),"① Nuage brut\n(sans classification)")
    sp(axes[1],X2d,[COLORS[k] for k in km.labels_],
       f"② K-Means (acc train={acc*100:.1f}%)",c2d=c_2d,leg=CLASSES)
    sp(axes[2],X2d,[COLORS[l] for l in labels],"③ Vrais labels\n(référence)",leg=CLASSES)
    plt.tight_layout()
    plt.savefig("etape6_nuages_3phases.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape6_nuages_3phases.png")

    # Correct vs Incorrect
    fig2,ax2=plt.subplots(figsize=(9,7)); fig2.patch.set_facecolor("#1A1A2E"); ax2.set_facecolor("#16213E")
    ax2.scatter(X2d[correct,0],X2d[correct,1],c="#2ECC71",s=32,alpha=0.8,edgecolors="none",
                label=f"Correct ({correct.sum()})")
    ax2.scatter(X2d[~correct,0],X2d[~correct,1],c="#E74C3C",s=32,alpha=0.8,edgecolors="none",
                label=f"Incorrect ({(~correct).sum()})")
    ax2.scatter(c_2d[:,0],c_2d[:,1],marker="*",s=350,c="white",edgecolors="#2C3E50",
                linewidths=1.5,zorder=10,label="Centroids")
    ax2.set_title("Correct ✔ vs Incorrect ✘ (K-Means)",color="white",fontsize=12,fontweight="bold")
    ax2.set_xlabel("PC1",color="white"); ax2.set_ylabel("PC2",color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=10,facecolor="#1A1A2E",labelcolor="white")
    ax2.grid(True,alpha=0.2,ls="--",color="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("etape6b_correct_vs_incorrect.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape6b_correct_vs_incorrect.png")


# ═══════════════════════════════════════════════════════════════════
#  ÉVALUATION COMPLÈTE : 4 MODÈLES CÔTE À CÔTE
# ═══════════════════════════════════════════════════════════════════

def confusion_matrix_manual(true,pred,k=N_CLUSTERS):
    cm=np.zeros((k,k),dtype=int)
    for t,p in zip(true,pred): cm[t,p]+=1
    return cm

def roc_curve_manual(y_true,y_score,n_thresh=200):
    """Courbe ROC from scratch."""
    thresholds=np.linspace(0,1,n_thresh)
    tprs,fprs=[],[]
    for t in thresholds:
        pred=(y_score>=t).astype(int)
        # classe positive = no_tumor (1) vs tumor (0)
        # on inverse pour ROC "tumor detection"
        pred_tumor=(y_score<t).astype(int)
        tp=((pred_tumor==1)&(y_true==0)).sum()
        fp=((pred_tumor==1)&(y_true==1)).sum()
        tn=((pred_tumor==0)&(y_true==1)).sum()
        fn=((pred_tumor==0)&(y_true==0)).sum()
        tpr=tp/(tp+fn+1e-8); fpr=fp/(fp+tn+1e-8)
        tprs.append(tpr); fprs.append(fpr)
    # AUC via trapèze
    fprs_arr=np.array(fprs); tprs_arr=np.array(tprs)
    idx=np.argsort(fprs_arr)
    auc=float(np.trapz(tprs_arr[idx],fprs_arr[idx]))
    return fprs_arr[idx],tprs_arr[idx],abs(auc)

def visualize_full_evaluation(km,X_train_norm,y_train,X_test_norm,y_test,
                               best_map,supervised_results,features_train,images_train):
    print("\n" + "═"*65)
    print("  ÉTAPE 7 — Évaluation complète (4 modèles)")
    print("═"*65)

    # K-Means predictions
    km_train_raw=km.labels_; km_test_raw=km.predict(X_test_norm)
    mapped_train=np.array([best_map[l] for l in km_train_raw])
    mapped_test =np.array([best_map[l] for l in km_test_raw])
    acc_km_train=(mapped_train==y_train).mean()
    acc_km_test =(mapped_test ==y_test ).mean()
    # Probabilité approchée pour ROC K-Means (distance normalisée)
    dists=np.array([[np.sum((x-c)**2) for c in km.centroids] for x in X_test_norm])
    km_prob=dists[:,0]/(dists.sum(axis=1)+1e-8)  # prob d'être dans cluster 0

    print(f"\n  {'Modèle':<18} {'Acc Train':>10} {'Acc Test':>10}")
    print(f"  {'─'*42}")
    print(f"  {'K-Means':<18} {acc_km_train*100:9.1f}%  {acc_km_test*100:9.1f}%")
    for name,res in supervised_results.items():
        if name.startswith("__"): continue
        print(f"  {name:<18} {'—':>9}   {res['acc']*100:9.1f}%")

    # ═══ FIGURE 1 : Matrices de confusion ═══
    all_models = {"K-Means": (mapped_test, None)}
    for name,res in supervised_results.items():
        if not name.startswith("__"):
            all_models[name] = (res["y_pred"], res.get("y_prob"))

    n_models = len(all_models)
    fig,axes=plt.subplots(1,n_models,figsize=(6*n_models,6))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("Matrices de confusion — Test Set (4 modèles)",
                 fontsize=14,fontweight="bold",color="white")

    for ax,(name,(y_pred,_)) in zip(axes,all_models.items()):
        cm=confusion_matrix_manual(y_test,y_pred); acc=(y_pred==y_test).mean()
        cmap="Blues" if name=="K-Means" else "Greens"
        ax.set_facecolor("#16213E")
        im=ax.imshow(cm,cmap=cmap)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        ax.set_xticks(range(N_CLUSTERS)); ax.set_yticks(range(N_CLUSTERS))
        ax.set_xticklabels(CLASSES,rotation=30,ha="right",color="white",fontsize=8)
        ax.set_yticklabels(CLASSES,color="white",fontsize=8)
        ax.set_xlabel("Prédit",color="white"); ax.set_ylabel("Réel",color="white")
        title_color="#F39C12" if name=="K-Means" else MODEL_COLORS.get(name,"white")
        ax.set_title(f"{name}\nAcc={acc*100:.1f}%",color=title_color,fontweight="bold")
        cm_n=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-8)
        for i in range(N_CLUSTERS):
            for j in range(N_CLUSTERS):
                ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",fontweight="bold",
                        fontsize=13,color="white" if cm_n[i,j]>0.4 else "#2C3E50")

    plt.tight_layout()
    plt.savefig("etape7_confusion_matrices.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape7_confusion_matrices.png")

    # ═══ FIGURE 2 : Courbes ROC ═══
    fig,ax=plt.subplots(figsize=(9,7))
    fig.patch.set_facecolor("#1A1A2E"); ax.set_facecolor("#16213E")
    # K-Means ROC
    fpr,tpr,auc=roc_curve_manual(y_test,km_prob)
    ax.plot(fpr,tpr,color=MODEL_COLORS["K-Means"],lw=2.5,
            label=f"K-Means (AUC={auc:.2f})",ls="--")
    # Modèles supervisés
    for name,res in supervised_results.items():
        if name.startswith("__"): continue
        if res.get("y_prob") is not None:
            fpr,tpr,auc=roc_curve_manual(y_test,1-res["y_prob"])
            ax.plot(fpr,tpr,color=MODEL_COLORS.get(name,"white"),lw=2.5,
                    label=f"{name} (AUC={auc:.2f})")
    ax.plot([0,1],[0,1],"w--",lw=1.2,alpha=0.4,label="Aléatoire (AUC=0.50)")
    ax.fill_between([0,1],[0,1],alpha=0.05,color="white")
    ax.set_xlabel("Taux de Faux Positifs (FPR)",color="white",fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (TPR / Recall)",color="white",fontsize=12)
    ax.set_title("Courbes ROC — Comparaison 4 modèles\n(Détection de tumeur)",
                 color="white",fontsize=13,fontweight="bold")
    ax.legend(fontsize=10,facecolor="#1A1A2E",labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(True,alpha=0.2,ls="--",color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("etape7_roc_curves.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape7_roc_curves.png")

    # ═══ FIGURE 3 : Comparaison accuracy + Feature Importance ═══
    fig,(ax_cmp,ax_fi)=plt.subplots(1,2,figsize=(18,7))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("Comparaison des modèles + Feature Importance (RF)",
                 fontsize=14,fontweight="bold",color="white")

    # Comparaison accuracy test
    ax_cmp.set_facecolor("#16213E")
    names_cmp=["K-Means"]+[n for n in supervised_results if not n.startswith("__")]
    accs_cmp=[acc_km_test]+[supervised_results[n]["acc"] for n in names_cmp[1:]]
    bar_cols=[MODEL_COLORS.get(n,"#666") for n in names_cmp]
    bars=ax_cmp.bar(names_cmp,accs_cmp,color=bar_cols,edgecolor="white",width=0.55)
    ax_cmp.set_ylim(0,1.15); ax_cmp.set_ylabel("Accuracy (Test Set)",color="white",fontsize=12)
    ax_cmp.set_title("Accuracy sur le test set\n(validation fiable)",color="white",fontweight="bold")
    ax_cmp.tick_params(colors="white")
    ax_cmp.grid(True,alpha=0.2,ls="--",color="white",axis='y')
    for sp in ax_cmp.spines.values(): sp.set_edgecolor("#444")
    for bar,acc in zip(bars,accs_cmp):
        ax_cmp.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                    f"{acc*100:.1f}%",ha="center",va="bottom",color="white",
                    fontweight="bold",fontsize=12)
    ax_cmp.axhline(y=0.5,color="white",ls="--",lw=1.2,alpha=0.5,label="Aléatoire")
    ax_cmp.legend(fontsize=9,facecolor="#1A1A2E",labelcolor="white")

    # Feature importance (Random Forest)
    ax_fi.set_facecolor("#16213E")
    fi=supervised_results["__rf_importance__"]
    idx_sorted=np.argsort(fi)
    fi_colors=["#F39C12" if i>=10 else "#3498DB" for i in idx_sorted]
    ax_fi.barh([FEATURE_NAMES[i] for i in idx_sorted],fi[idx_sorted],
               color=fi_colors,edgecolor="#444")
    ax_fi.set_xlabel("Importance (Random Forest)",color="white",fontsize=11)
    ax_fi.set_title("Feature Importance\n(🟠 nouvelles features)",color="white",fontweight="bold")
    ax_fi.tick_params(colors="white")
    ax_fi.grid(True,alpha=0.2,ls="--",color="white",axis='x')
    for sp in ax_fi.spines.values(): sp.set_edgecolor("#444")
    for i,(bar,val) in enumerate(zip(ax_fi.patches,fi[idx_sorted])):
        ax_fi.text(val+0.002,bar.get_y()+bar.get_height()/2,
                   f"{val:.3f}",va="center",color="white",fontsize=7)

    plt.tight_layout()
    plt.savefig("etape7_comparaison_modeles.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape7_comparaison_modeles.png")

    # ═══ FIGURE 4 : Galerie prédictions K-Means ═══
    n_per_class=max(1,len(y_train)//(N_CLUSTERS*2)); n_cols=10
    idxs_all=[]
    for ci in range(N_CLUSTERS): idxs_all.extend(np.where(y_train==ci)[0][:n_per_class].tolist())
    n_total=len(idxs_all); n_rows=max(1,int(np.ceil(n_total/n_cols)))
    fig2,axes2=plt.subplots(n_rows,n_cols,figsize=(n_cols*2.2,n_rows*2.4))
    if n_rows==1: axes2=axes2.reshape(1,-1)
    fig2.patch.set_facecolor("#0D0D0D")
    fig2.suptitle(f"Galerie prédictions K-Means — Train set ({n_total} images)",
                  fontsize=13,fontweight="bold",color="white")
    for ai,img_idx in enumerate(idxs_all):
        r2,c2=ai//n_cols,ai%n_cols; ax=axes2[r2][c2]
        ax.imshow(images_train[img_idx],cmap="gray")
        tc=y_train[img_idx]; pc=mapped_train[img_idx]; ok=tc==pc
        for sp in ax.spines.values():
            sp.set_edgecolor("#2ECC71" if ok else "#E74C3C"); sp.set_linewidth(3)
        ax.set_title(f"V:{CLASSES[tc][:5]}\nP:{CLASSES[pc][:5]}",
                     fontsize=6,color="#2ECC71" if ok else "#E74C3C",fontweight="bold")
        ax.axis("off")
    for ai in range(n_total,n_rows*n_cols): axes2[ai//n_cols][ai%n_cols].axis("off")
    fig2.legend(handles=[mpatches.Patch(color="#2ECC71",label="Correct"),
                          mpatches.Patch(color="#E74C3C",label="Incorrect")],
                loc="lower center",ncol=2,fontsize=11,facecolor="#1A1A2E",labelcolor="white",
                bbox_to_anchor=(0.5,0.01))
    plt.tight_layout(); plt.subplots_adjust(bottom=0.05)
    plt.savefig("etape7_galerie.png",dpi=110,bbox_inches="tight",facecolor="#0D0D0D")
    plt.close(); print("  → Sauvegardé : etape7_galerie.png")

    # ═══ FIGURE 5 : Radar features ═══
    fig5=plt.figure(figsize=(10,8)); fig5.patch.set_facecolor("#1A1A2E")
    ax_r=fig5.add_subplot(111,polar=True)
    n_feat=len(FEATURE_NAMES)
    angles=np.linspace(0,2*np.pi,n_feat,endpoint=False)
    angles=np.concatenate([angles,[angles[0]]])
    Xn,_,_=zscore(features_train)
    Xn=(Xn-Xn.min())/(Xn.max()-Xn.min()+1e-8)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(FEATURE_NAMES,color="white",fontsize=7)
    ax_r.tick_params(colors="white"); ax_r.set_facecolor("#16213E")
    for ci in range(N_CLUSTERS):
        vals=Xn[y_train==ci].mean(axis=0).tolist()+[Xn[y_train==ci].mean(axis=0)[0]]
        ax_r.plot(angles,vals,color=COLORS[ci],lw=2,label=CLASSES[ci])
        ax_r.fill(angles,vals,color=COLORS[ci],alpha=0.12)
    ax_r.set_title("Profil moyen des 13 features par classe",color="white",fontweight="bold",pad=20)
    ax_r.legend(loc="upper right",bbox_to_anchor=(1.4,1.15),fontsize=9,
                facecolor="#1A1A2E",labelcolor="white")
    plt.tight_layout()
    plt.savefig("etape7_radar_features.png",dpi=130,bbox_inches="tight",facecolor="#1A1A2E")
    plt.close(); print("  → Sauvegardé : etape7_radar_features.png")

    return acc_km_test,{n:r["acc"] for n,r in supervised_results.items() if not n.startswith("__")}


# ═══════════════════════════════════════════════════════════════════
#  VARIABLES GLOBALES
# ═══════════════════════════════════════════════════════════════════
labels_global      = None
images_proc_global = None
km_global          = None
mu_global          = None
sig_global         = None
best_map_global    = None
rf_global          = None   # Random Forest pour prédiction


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main(dataset_path):
    global labels_global,images_proc_global
    global km_global,mu_global,sig_global,best_map_global,rf_global

    print("\n╔" + "═"*65 + "╗")
    print("║   SYSTÈME CAD — Classification MRI Cérébrale                   ║")
    print("║   K-Means (from scratch) + Random Forest + SVM + LR            ║")
    print("╚" + "═"*65 + "╝")

    # 1. Chargement
    images_raw,labels=load_images(dataset_path)

    # 2. Pré-traitement
    images_proc=preprocess_all(images_raw,labels)

    # 3. Segmentation (visualisation)
    visualize_segmentation(images_proc,labels)

    # 4. Extraction 13 features
    print("\n" + "═"*65)
    print("  ÉTAPE 4 — Extraction des 13 features")
    print("═"*65)
    features,valid_indices=extract_features(images_proc,labels)
    labels_valid=labels[valid_indices]; images_proc_valid=images_proc[valid_indices]
    print(f"\n  ✔  {len(features)} images valides | {features.shape[1]} features")

    # 5. Train/Test split
    print("\n" + "═"*65)
    print("  ÉTAPE 5 — Split Train/Test (80% / 20%)")
    print("═"*65)
    train_idx,test_idx=train_test_split_stratified(features,labels_valid)
    X_train=features[train_idx]; y_train=labels_valid[train_idx]
    X_test =features[test_idx];  y_test =labels_valid[test_idx]
    images_train=images_proc_valid[train_idx]

    # Normalisation (fit sur train uniquement)
    X_train_norm,mu,sig=zscore(X_train)
    X_test_norm=(X_test-mu)/(sig+1e-8)
    mu_global=mu; sig_global=sig

    # Globals
    labels_global=y_train; images_proc_global=images_train

    # 5a. Coude
    elbow(X_train_norm)

    # 5b. K-Means final
    print("\n" + "═"*65)
    print(f"  ÉTAPE 5b — K-Means from scratch (k={N_CLUSTERS}, n_init={KMEANS_NINIT})")
    print("═"*65)
    km=KMeans(k=N_CLUSTERS,max_iter=KMEANS_ITER,n_init=KMEANS_NINIT)
    km.fit(X_train_norm); km_global=km
    acc_km_train,best_map=align_clusters(km.labels_,y_train); best_map_global=best_map

    # 6. Nuages de points
    visualize_scatter_3phases(km,X_train_norm,y_train,best_map)

    # 6b. Modèles supervisés
    supervised_results=train_supervised_models(X_train_norm,y_train,X_test_norm,y_test)
    rf_global=supervised_results["Random Forest"]["clf"]

    # 7. Évaluation complète
    acc_km,accs_sup=visualize_full_evaluation(
        km,X_train_norm,y_train,X_test_norm,y_test,
        best_map,supervised_results,X_train,images_train)

    # 8. Résumé final
    print("\n" + "═"*65)
    print("  RÉSUMÉ FINAL — Comparaison des 4 modèles (Test Set)")
    print("═"*65)
    print(f"\n  {'Modèle':<20} {'Accuracy Test':>14}  {'Remarque'}")
    print(f"  {'─'*60}")
    print(f"  {'K-Means':<20} {acc_km*100:13.1f}%  (non supervisé — baseline)")
    for name,acc in accs_sup.items():
        best_tag=" ← MEILLEUR" if acc==max(accs_sup.values()) else ""
        print(f"  {name:<20} {acc*100:13.1f}%  (supervisé){best_tag}")

    print("\n" + "═"*65)
    print("  ✅  Fichiers générés :")
    for f in ["etape2_pretraitement.png","etape3_segmentation_tumor.png",
              "etape3_segmentation_sain.png","etape3_segmentation_classes.png",
              "etape5a_coude.png","etape6_nuages_3phases.png",
              "etape6b_correct_vs_incorrect.png","etape7_confusion_matrices.png",
              "etape7_roc_curves.png","etape7_comparaison_modeles.png",
              "etape7_galerie.png","etape7_radar_features.png"]:
        print(f"  → {f}")
    print("═"*65)

    return km,mu,sig,rf_global


# ═══════════════════════════════════════════════════════════════════
#  PRÉDICTION NOUVELLE IMAGE (utilise le meilleur modèle = RF)
# ═══════════════════════════════════════════════════════════════════

def predict_new_image(image_path, km, mu, sig, rf=None):
    print("\n" + "═"*65)
    print("  PRÉDICTION — Nouvelle image MRI")
    print("═"*65)

    raw  = np.array(Image.open(image_path).convert("L").resize((IMG_SIZE,IMG_SIZE)))
    proc = preprocess(raw)

    feat_vec,valid_idx=extract_features(np.array([proc]),np.array([0]))
    if len(feat_vec)==0:
        print("  ⚠️  Segmentation insuffisante — prédiction impossible.")
        return None, None

    feat_norm=(feat_vec-mu)/(sig+1e-8)

    # K-Means prediction
    distances=np.linalg.norm(feat_norm-km.centroids,axis=1)
    cluster_id=int(np.argmin(distances))
    _,best_map_v=align_clusters(km.labels_,labels_global)
    km_pred=CLASSES[best_map_v[cluster_id]]

    # Random Forest prediction (si disponible)
    rf_pred=None
    if rf is not None:
        rf_class=rf.predict(feat_norm)[0]
        rf_prob =rf.predict_proba(feat_norm)[0]
        rf_pred =CLASSES[rf_class]
        rf_conf =rf_prob.max()*100

    # Choix du diagnostic final = RF si disponible, sinon K-Means
    final_pred = rf_pred if rf_pred is not None else km_pred
    diagnosis  = "⚠️ TUMEUR DÉTECTÉE" if final_pred=="tumor" else "✅ AUCUNE TUMEUR"

    print(f"\n  Image        : {os.path.basename(image_path)}")
    print(f"  K-Means      → {km_pred}")
    if rf_pred: print(f"  Random Forest→ {rf_pred} (confiance : {rf_conf:.1f}%)")
    print(f"\n  ╔{'═'*43}╗")
    print(f"  ║  DIAGNOSTIC : {diagnosis:<27}║")
    print(f"  ╚{'═'*43}╝")

    final_mask = segment_tumor_chanvese(proc)

    n_panels=5 if rf_pred else 4
    fig,axes=plt.subplots(1,n_panels,figsize=(5*n_panels,5))
    fig.patch.set_facecolor("#0D0D0D")
    panels=[(raw,"gray","Image brute"),(proc,"gray","Pré-traitée"),
            (final_mask.astype(float),"hot","Masque (K-Means)"),
            (overlay_tumor(proc,final_mask),None,f"K-Means\n→ {km_pred}")]
    if rf_pred:
        color=(1,0,0) if rf_pred=="tumor" else (0,0.8,0.2)
        panels.append((overlay_tumor(proc,final_mask,color=color),None,
                        f"RF\n→ {rf_pred} ({rf_conf:.0f}%)"))
    for ax,(data,cm,lbl) in zip(axes,panels):
        ax.imshow(data,cmap=cm) if cm else ax.imshow(data)
        ax.set_title(lbl,color="white",fontsize=10,fontweight="bold"); ax.axis("off")

    if rf_pred:
        ax_bar=fig.add_axes([0.92,0.15,0.06,0.65])
        ax_bar.set_facecolor("#16213E")
        bar_cols=[COLORS[i] for i in range(N_CLUSTERS)]
        ax_bar.barh(CLASSES,rf_prob,color=bar_cols,edgecolor="white")
        ax_bar.set_xlim(0,1); ax_bar.tick_params(colors="white",labelsize=7)
        ax_bar.set_title("RF\nprob",color="white",fontsize=7)
        for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")

    plt.savefig("prediction.png",dpi=130,bbox_inches="tight",facecolor="#0D0D0D")
    plt.close(); print("  → Sauvegardé : prediction.png")
    return final_pred, rf_prob if rf_pred else None


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = r"C:\Users\pc\Desktop\CNN\Training"
    km, mu, sig, rf = main(dataset_path)

    new_image_path = r"C:\Users\pc\Desktop\CNN\Training\no_tumor\image (5).jpg"
    predict_new_image(new_image_path, km, mu, sig, rf)
