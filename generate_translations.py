"""
One-time script: reads disease_info.json, translates all text fields
to hi/ta/te/ml using deep-translator, saves back to disease_info.json.
Run once, then delete this script.
"""
import json
from deep_translator import GoogleTranslator

LANGS = {"hi": "hi", "ta": "ta", "te": "te", "ml": "ml"}

def t(text, lang):
    if not text:
        return text
    try:
        return GoogleTranslator(source='en', target=lang).translate(text)
    except:
        return text

with open("disease_info.json", "r") as f:
    data = json.load(f)

for disease_key, info in data.items():
    print(f"Translating: {disease_key}")
    old_trans = info.get("translations", {})
    new_trans = {}
    
    for lang_code in LANGS:
        entry = {}
        # disease name — keep existing if available
        if isinstance(old_trans.get(lang_code), str):
            entry["disease_name"] = old_trans[lang_code]
        else:
            entry["disease_name"] = t(info.get("disease_name", ""), lang_code)
        
        # description
        entry["description"] = t(info.get("description", ""), lang_code)
        
        # cause
        entry["cause"] = t(info.get("cause", ""), lang_code)
        
        # symptoms
        syms = info.get("symptoms", [])
        entry["symptoms"] = [t(s, lang_code) for s in syms]
        
        # recommendation
        rec = info.get("recommendation", {})
        if isinstance(rec, dict):
            entry["recommendation"] = {
                k: t(v, lang_code) for k, v in rec.items()
            }
        elif isinstance(rec, str):
            entry["recommendation"] = t(rec, lang_code)
        
        new_trans[lang_code] = entry
    
    info["translations"] = new_trans

with open("disease_info.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Done! Translations saved to disease_info.json")
