from app.recommender import load_data, recommend
from app.nlp_pipeline import predict_intent, predict_entities
import re
import string
import sys
from typing import Optional, Dict, Any
import pandas as pd

# ==========================================
# CONVERSATION MEMORY
# ==========================================
conversation_memory: Dict[str, Any] = {
    "last_query": None,
    "last_results": None,
    "last_params": None
}

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================
def preprocess_text(text):
    text = text.lower() # Lowercasing

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove emojis (basic pattern)
    emoji_pattern = re.compile("["
                           "\U0001F600-\U0001F64F"  # emoticons
                           "\U0001F300-\U0001F5FF"  # symbols & pictographs
                           "\U0001F680-\U0001F6FF"  # transport & map symbols
                           "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "\U00002702-\U000027B0"
                           "\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove special characters (punctuation and other non-alphanumeric except space)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove any remaining non-alphanumeric chars not covered by string.punctuation

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def format_idr(usd_price):
    """Format harga USD ke Rupiah"""
    idr_price = usd_price * 16000
    return f"Rp {idr_price:,.0f}".replace(",", ".")

df = load_data()

def extract_laptop_name(entities: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract laptop brand and model from NER entities, returns (brand, model)"""
    # Combine BRAND tokens
    brand_tokens = entities.get("B-BRAND", []) + entities.get("I-BRAND", [])
    # Filter out BERT special tokens
    brand_tokens = [t for t in brand_tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    brand = "".join(brand_tokens).replace("##", "") if brand_tokens else None
    
    # Combine MODEL tokens
    model_tokens = entities.get("B-MODEL", []) + entities.get("I-MODEL", [])
    # Filter out BERT special tokens
    model_tokens = [t for t in model_tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    model = "".join(model_tokens).replace("##", "") if model_tokens else None
    
    print(f"[DEBUG] Brand tokens: {brand_tokens} → '{brand}'")
    print(f"[DEBUG] Model tokens: {model_tokens} → '{model}'")
    
    return brand, model

def extract_params(text: str, entities: dict):
    """Extract recommendation parameters from text and entities"""
    params = {}
    
    # Extract BRAND
    if "B-BRAND" in entities or "I-BRAND" in entities:
        brand_tokens = entities.get("B-BRAND", []) + entities.get("I-BRAND", [])
        # Filter out BERT special tokens
        brand_tokens = [t for t in brand_tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
        brand = "".join(brand_tokens).replace("##", "")
        if brand:
            params["brand"] = brand
    
    # Extract USAGE
    if "B-USAGE" in entities or "I-USAGE" in entities:
        usage_tokens = entities.get("B-USAGE", []) + entities.get("I-USAGE", [])
        usage_text = "".join(usage_tokens).replace("##", "").lower()
        
        if "gaming" in usage_text or "game" in usage_text:
            params["usage"] = "gaming"
        elif "coding" in usage_text or "programming" in usage_text or "kuliah" in usage_text:
            params["usage"] = "coding"
        elif "editing" in usage_text or "edit" in usage_text or "desain" in usage_text:
            params["usage"] = "editing"
    
    # Extract BUDGET (convert juta to millions)
    if "B-BUDGET" in entities or "I-BUDGET" in entities:
        budget_tokens = entities.get("B-BUDGET", []) + entities.get("I-BUDGET", [])
        budget_text = "".join(budget_tokens).replace("##", "")
        # Extract numbers
        numbers = re.findall(r'\d+', budget_text)
        if numbers:
            budget_val = int(numbers[0])
            # If text contains "juta" or "jt", multiply by 1000000
            if "juta" in text.lower() or "jt" in text.lower():
                budget_val *= 1000000
            params["budget"] = budget_val
    
    # Extract SCREEN_SIZE
    if "B-SCREEN_SIZE" in entities or "I-SCREEN_SIZE" in entities:
        screen_tokens = entities.get("B-SCREEN_SIZE", []) + entities.get("I-SCREEN_SIZE", [])
        screen_text = "".join(screen_tokens).replace("##", "")
        # Extract number
        numbers = re.findall(r'\d+\.?\d*', screen_text)
        if numbers:
            params["screen_size"] = float(numbers[0])
    
    # Extract RAM
    if "B-RAM" in entities or "I-RAM" in entities:
        ram_tokens = entities.get("B-RAM", []) + entities.get("I-RAM", [])
        ram_text = "".join(ram_tokens).replace("##", "")
        numbers = re.findall(r'\d+', ram_text)
        if numbers:
            params["ram"] = int(numbers[0])
    
    # Extract STORAGE
    if "B-STORAGE" in entities or "I-STORAGE" in entities:
        storage_tokens = entities.get("B-STORAGE", []) + entities.get("I-STORAGE", [])
        storage_text = "".join(storage_tokens).replace("##", "")
        numbers = re.findall(r'\d+', storage_text)
        if numbers:
            params["storage"] = int(numbers[0])
    
    # Extract TOUCHSCREEN
    if "B-TOUCHSCREEN" in entities or "I-TOUCHSCREEN" in entities:
        touchscreen_tokens = entities.get("B-TOUCHSCREEN", []) + entities.get("I-TOUCHSCREEN", [])
        touchscreen_text = "".join(touchscreen_tokens).replace("##", "").lower()
        
        # Check if user wants touchscreen
        if "touch" in touchscreen_text or "sentuh" in touchscreen_text:
            params["touchscreen"] = True
    
    # Also check in plain text for touchscreen keywords
    if "touchscreen" in text.lower() or "layar sentuh" in text.lower() or "touch screen" in text.lower():
        params["touchscreen"] = True
    
    return params

def chatbot_reply(user_text: str):
    intent = predict_intent(user_text)
    print(f"[DEBUG] Intent: {intent}")
    
    if intent == "goodbye":
        return("Terima kasih! Semoga harimu menyenangkan. 👋")

    elif intent == "greeting":
        return("Halo! Ada yang bisa saya bantu? Mau cari laptop jenis apa?")

    elif intent == "fallback":
        return("Maaf, saya kurang mengerti. Bisa jelaskan lebih detail spesifikasi laptop yang dicari?")

    elif intent == "ask_recommendation":
        # Extract entities and parameters
        entities = predict_entities(user_text)
        print(f"[DEBUG] Entities: {entities}")
        params = extract_params(user_text, entities)
        print(f"[DEBUG] Extracted Params: {params}")
        
        # Get recommendations with extracted parameters
        results = recommend(df, **params)
        
        if results.empty:
            return "Maaf, tidak ada laptop yang sesuai dengan kriteria Anda. Coba ubah kriteria pencarian."
        
        # Store in memory for follow-up questions
        conversation_memory["last_query"] = user_text
        conversation_memory["last_results"] = results.copy()
        conversation_memory["last_params"] = params
        
        results = format_results(results)
        return "Berikut rekomendasi laptop:\n" + results.to_string(index=False)

    # --- PERBAIKAN LOGIKA ASK_SPECS (with NER) ---
    elif intent == "ask_specs":
        # Use NER to extract laptop brand/name
        entities = predict_entities(user_text)
        print(f"[DEBUG] Entities for ask_specs: {entities}")
        brand, model = extract_laptop_name(entities)
        print(f"[DEBUG] Extracted brand: {brand}, model: {model}")
        
        # Build search filter
        filtered_laptop = df.copy()
        
        if brand:
            filtered_laptop = filtered_laptop[
                filtered_laptop['Brand'].str.contains(brand, case=False, na=False)
            ]
        
        if model:
            filtered_laptop = filtered_laptop[
                filtered_laptop['Model'].str.contains(model, case=False, na=False)
            ]
        
        # Fallback to regex if NER doesn't find anything
        if not brand and not model:
            match = re.search(r'(?:spesifikasi|spek)\s+([\w\s]+)|([\w\s]+?)\s+(?:gimana|speknya|itu)', user_text, re.IGNORECASE)
            if match:
                search_term = match.group(1) if match.group(1) else match.group(2)
                if search_term and len(search_term.strip()) > 2:
                    filtered_laptop = df[
                        df['Brand'].str.contains(search_term.strip(), case=False, na=False) |
                        df['Model'].str.contains(search_term.strip(), case=False, na=False)
                    ]
        
        if not filtered_laptop.empty:
            row = filtered_laptop.iloc[0]
            specs = f"Spesifikasi {row['Brand']} {row['Model']}:\n"
            specs += f"   • CPU: {row['CPU']}\n"
            specs += f"   • RAM: {row['RAM']}GB | Storage: {row['Storage']}GB\n"
            specs += f"   • GPU: {row['GPU']}\n"
            specs += f"   • Screen: {row['Screen']}\" | Touch: {row['Touch']}\n"
            specs += f"   • Harga: {format_idr(row['Final Price'])}"
            return specs
        else:
            search_desc = f"{brand} {model}".strip() if brand or model else "laptop tersebut"
            return f"Maaf, saya tidak menemukan laptop '{search_desc}'."

    # --- PERBAIKAN LOGIKA ASK_PRICE (with NER) ---
    elif intent == "ask_price":
        # Use NER to extract laptop brand/name
        entities = predict_entities(user_text)
        print(f"[DEBUG] Entities for ask_price: {entities}")
        brand, model = extract_laptop_name(entities)
        print(f"[DEBUG] Extracted brand: {brand}, model: {model}")
        
        # Build search filter
        filtered_laptop = df.copy()
        
        if brand:
            filtered_laptop = filtered_laptop[
                filtered_laptop['Brand'].str.contains(brand, case=False, na=False)
            ]
        
        if model:
            filtered_laptop = filtered_laptop[
                filtered_laptop['Model'].str.contains(model, case=False, na=False)
            ]
        
        # Fallback to regex if NER doesn't find anything
        if not brand and not model:
            match = re.search(r'(?:harga)\s+([\w\s]+)|([\w\s]+?)\s+(?:berapa|harganya)', user_text, re.IGNORECASE)
            if match:
                search_term = match.group(1) if match.group(1) else match.group(2)
                if search_term and len(search_term.strip()) > 2:
                    filtered_laptop = df[
                        df['Brand'].str.contains(search_term.strip(), case=False, na=False) |
                        df['Model'].str.contains(search_term.strip(), case=False, na=False)
                    ]
        
        if not filtered_laptop.empty:
            row = filtered_laptop.iloc[0]
            harga = format_idr(row['Final Price'])
            return f"Harga {row['Brand']} {row['Model']} sekitar {harga}."
        else:
            search_desc = f"{brand} {model}".strip() if brand or model else "laptop tersebut"
            return f"Maaf, harga laptop '{search_desc}' tidak ditemukan."

    elif intent == "clarify_requirement":
        # Use conversation memory to provide context-aware clarification
        if conversation_memory["last_params"]:
            params = conversation_memory["last_params"]
            clarification = "Saya sudah punya beberapa kriteria Anda:\n"
            
            if "brand" in params:
                clarification += f"   • Brand: {params['brand']}\n"
            if "usage" in params:
                clarification += f"   • Kegunaan: {params['usage']}\n"
            if "budget" in params:
                clarification += f"   • Budget: {format_idr(params['budget']/16000)}\n"
            if "ram" in params:
                clarification += f"   • RAM: {params['ram']}GB\n"
            
            clarification += "\nAda kriteria tambahan yang ingin ditambahkan? (misal: ukuran layar, touchscreen, dll.)"
            return clarification
        else:
            return "Baik, saya mengerti. Bisakah Anda memberikan lebih banyak detail tentang kebutuhan Anda? Misalnya, untuk keperluan apa laptop ini akan digunakan (gaming, kerja, desain, dll.), atau fitur spesifik lainnya?"

    else:
        return("Maaf, saya belum paham maksud Anda.")

def format_results(df):
    df = df.copy()
    # Convert USD → IDR for display
    df["Final Price (IDR)"] = df["Final Price"].apply(lambda x: format_idr(x))

    # Drop USD column for user-facing output
    df = df.drop(columns=["Final Price"])

    return df