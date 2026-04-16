# autosurf.py - Versione per Render (senza dotenv)

import os
import time
import requests
import numpy as np
import cv2
import json
from datetime import datetime
from supabase import create_client
from datasets import load_dataset
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "errori"
DATASET_REPO = "zenadazurli/easyhits4u-dataset"

os.makedirs(ERRORI_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# ================ SUPABASE CONFIG (da variabili d'ambiente Render) =====================
SUPABASE_URL = os.environ.get("COOKIES_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("COOKIES_SUPABASE_KEY")
ACCOUNT_NAME = os.environ.get("ACCOUNT_NAME", "nicoladellaaziendavinicola")

# ... resto del codice identico ...