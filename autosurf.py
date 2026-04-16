# autosurf.py - Scarica dataset_speed.npz da Hugging Face

import os
import sys
import time
import requests
import numpy as np
import cv2
import json
from datetime import datetime
from supabase import create_client
from huggingface_hub import hf_hub_download
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================ DEBUG AVVIO =====================
print("=" * 50, flush=True)
print("🚀 AVVIO AUTOSURF", flush=True)
print("=" * 50, flush=True)

# Verifica variabili d'ambiente
supabase_url = os.environ.get("COOKIES_SUPABASE_URL")
supabase_key = os.environ.get("COOKIES_SUPABASE_KEY")
account_name = os.environ.get("ACCOUNT_NAME", "nicoladellaaziendavinicola")
hf_token = os.environ.get("HF_TOKEN")  # Opzionale, per rate limit migliori

print(f"COOKIES_SUPABASE_URL: {'✅' if supabase_url else '❌ MANCANTE'}", flush=True)
print(f"COOKIES_SUPABASE_KEY: {'✅' if supabase_key else '❌ MANCANTE'}", flush=True)
print(f"ACCOUNT_NAME: {account_name}", flush=True)
print(f"HF_TOKEN: {'✅' if hf_token else '❌ NON SET (rate limit più basso)'}", flush=True)

if not supabase_url or not supabase_key:
    print("❌ ERRORE: Variabili d'ambiente mancanti!", flush=True)
    sys.exit(1)

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "errori"
DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "dataset_speed.npz")
HF_REPO = "zenadazurli/easyhits4u-dataset"
HF_FILENAME = "dataset_speed.npz"

os.makedirs(ERRORI_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ================ GLOBALS =====================
X_fast = None
y_fast = None
classes_fast = None
current_cookie_string = None

# ================ LOG =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ================ DOWNLOAD DATASET DA HUGGING FACE =====================
def download_dataset():
    """Scarica dataset_speed.npz da Hugging Face"""
    
    # Se già esiste, verifica se è valido
    if os.path.exists(DATASET_PATH):
        log(f"✅ Dataset già presente in {DATASET_PATH}")
        return True
    
    log(f"📥 Download dataset da Hugging Face: {HF_REPO}/{HF_FILENAME}")
    
    try:
        # Prova con token se fornito
        if hf_token:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILENAME,
                token=hf_token,
                local_dir=DATASET_DIR,
                local_dir_use_symlinks=False
            )
        else:
            # Senza token (rate limit più basso ma funziona)
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILENAME,
                local_dir=DATASET_DIR,
                local_dir_use_symlinks=False
            )
        
        log(f"✅ Dataset scaricato in {downloaded_path}")
        return True
        
    except Exception as e:
        log(f"❌ Errore download dataset: {e}")
        return False

# ================ CARICAMENTO DATASET =====================
def load_dataset():
    global X_fast, y_fast, classes_fast
    
    if not os.path.exists(DATASET_PATH):
        log(f"❌ Dataset non trovato: {DATASET_PATH}")
        return False
    
    log(f"📥 Caricamento dataset da {DATASET_PATH}")
    
    try:
        data = np.load(DATASET_PATH, allow_pickle=True)
        X_fast = data["X"].astype(np.float32)
        y_fast = data["y"].astype(np.int32)
        
        if "classes" in data.files:
            classes = list(np.array(data["classes"], dtype=object).tolist())
        else:
            unique = sorted(list(set(int(x) for x in y_fast.tolist())))
            classes = [str(c) for c in unique]
        
        classes_fast = {i: classes[i] for i in range(len(classes))}
        
        log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes)} classi")
        return True
        
    except Exception as e:
        log(f"❌ Errore caricamento dataset: {e}")
        return False

# ================ SUPABASE FUNCTIONS =====================
def get_cookie_from_supabase():
    try:
        supabase = create_client(supabase_url, supabase_key)
        resp = supabase.table('account_cookies')\
            .select('cookies_string', 'user_id', 'sesid')\
            .eq('account_name', account_name)\
            .eq('status', 'active')\
            .execute()
        
        if resp.data:
            return resp.data[0]['cookies_string']
        return None
    except Exception as e:
        log(f"❌ Errore lettura cookie: {e}")
        return None

def refresh_cookie():
    log("🔄 Attendo nuovo cookie...")
    for _ in range(30):
        time.sleep(10)
        cookie = get_cookie_from_supabase()
        if cookie and cookie != current_cookie_string:
            return cookie
    return None

# ================ FUNZIONI DI RICONOSCIMENTO =====================
def centra_figura(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()

    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])

    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])

    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

def predict(img_crop):
    global X_fast, y_fast, classes_fast
    
    if X_fast is None or img_crop is None or img_crop.size == 0:
        return None
    
    features = get_features(img_crop)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), "errore")

def crop_safe(img, coords):
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

# ================ SALVATAGGIO ERRORI =====================
def salva_errore(qpic, img, picmap, labels, chosen_idx, motivo, urlid=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(ERRORI_DIR, f"{timestamp}_{qpic}")
    os.makedirs(folder, exist_ok=True)
    
    full_path = os.path.join(folder, "full.jpg")
    cv2.imwrite(full_path, img)
    
    for i, p in enumerate(picmap):
        crop = crop_safe(img, p.get("coords", ""))
        if crop is not None and crop.size > 0:
            crop_path = os.path.join(folder, f"crop_{i+1}.jpg")
            cv2.imwrite(crop_path, crop)
    
    metadata = {
        "timestamp": timestamp,
        "qpic": qpic,
        "urlid": urlid,
        "motivo": motivo,
        "labels_predette": labels,
        "chosen_idx": chosen_idx,
    }
    
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    log(f"📁 Errore salvato in {folder}")

# ================ MAIN LOOP =====================
def main():
    global current_cookie_string
    
    log("=" * 50)
    log("🚀 EasyHits4U Autosurf")
    log("=" * 50)
    
    # Scarica dataset da Hugging Face
    if not download_dataset():
        log("❌ Impossibile scaricare il dataset")
        return
    
    # Carica dataset
    if not load_dataset():
        log("❌ Impossibile caricare il dataset")
        return
    
    # Ottieni cookie
    current_cookie_string = get_cookie_from_supabase()
    if not current_cookie_string:
        log("❌ Nessun cookie attivo trovato")
        return
    
    log("✅ Cookie ottenuto")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": current_cookie_string
    }
    session = requests.Session()
    session.headers.update(headers)
    
    captcha_counter = 0
    errori_consecutivi = 0
    MAX_ERRORI = 2
    
    while True:
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            
            if r.status_code != 200:
                time.sleep(5)
                continue
            
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])
            
            if not urlid or not qpic or not picmap:
                log("⚠️ Cookie scaduto")
                errori_consecutivi += 1
                new_cookie = refresh_cookie()
                if new_cookie:
                    current_cookie_string = new_cookie
                    session.headers.update({"Cookie": current_cookie_string})
                    errori_consecutivi = 0
                continue
            
            errori_consecutivi = 0
            
            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
            labels = [predict(c) for c in crops]
            log(f"📋 Labels: {labels}")
            
            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i
            
            if chosen_idx is None:
                log("❌ Nessun duplicato")
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
                errori_consecutivi += 1
                if errori_consecutivi >= MAX_ERRORI:
                    break
                time.sleep(seconds)
                continue
            
            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                verify=False
            )
            
            if resp.json().get("warning") == "wrong_choice":
                log("❌ Wrong choice")
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                errori_consecutivi += 1
                if errori_consecutivi >= MAX_ERRORI:
                    break
                continue
            
            captcha_counter += 1
            errori_consecutivi = 0
            log(f"✅ OK #{captcha_counter} - indice {chosen_idx}")
            
            time.sleep(2)
            
        except Exception as e:
            log(f"❌ Errore: {e}")
            errori_consecutivi += 1
            if errori_consecutivi >= MAX_ERRORI:
                break
            time.sleep(5)

if __name__ == "__main__":
    main()
