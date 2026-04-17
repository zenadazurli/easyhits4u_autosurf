# ================ MAIN LOOP MODIFICATO =====================
def main():
    global current_cookie_string
    
    log("=" * 50)
    log("🚀 EasyHits4U Autosurf")
    log("=" * 50)
    
    if not load_dataset_from_hf():
        log("❌ Impossibile proseguire senza dataset")
        return
    
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
    # RIMOSSO errori_consecutivi - ci fermiamo al primo errore!
    
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
            
            # Verifica cookie scaduto
            if not urlid or not qpic or not picmap:
                log("⚠️ Cookie scaduto")
                new_cookie = refresh_cookie()
                if new_cookie:
                    current_cookie_string = new_cookie
                    session.headers.update({"Cookie": current_cookie_string})
                    log("✅ Cookie aggiornato")
                continue
            
            # Scarica immagine
            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Riconosci figure
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
            labels = [predict(c) for c in crops]
            log(f"📋 Labels: {labels}")
            
            # Trova duplicati
            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        log(f"🎯 Duplicato: '{label}' posizioni {seen[label]}+{i}")
                        break
                    seen[label] = i
            
            # SE NESSUN DUPLICATO -> ERRORE -> STOP IMMEDIATO
            if chosen_idx is None:
                log("❌ NESSUN DUPLICATO - Errore riconoscimento")
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
                log("🛑 FERMO PER ANALISI ERRORI")
                log("   Controlla la cartella 'errori/' e rinomina i crop con l'etichetta corretta")
                break  # <--- STOP IMMEDIATO
            
            # Attendi e invia
            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                verify=False
            )
            
            # SE WRONG CHOICE -> ERRORE -> STOP IMMEDIATO
            if resp.json().get("warning") == "wrong_choice":
                log("❌ WRONG CHOICE - Errore riconoscimento")
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                log("🛑 FERMO PER ANALISI ERRORI")
                log("   Controlla la cartella 'errori/' e rinomina i crop con l'etichetta corretta")
                break  # <--- STOP IMMEDIATO
            
            # Successo!
            captcha_counter += 1
            log(f"✅ OK #{captcha_counter} - indice {chosen_idx}")
            
            time.sleep(2)
            
        except Exception as e:
            log(f"❌ Errore: {e}")
            # In caso di eccezione, salva se possibile
            try:
                salva_errore(qpic, img, picmap, labels, None, f"eccezione_{e}", urlid)
            except:
                pass
            log("🛑 FERMO PER ERRORE")
            break  # <--- STOP IMMEDIATO
