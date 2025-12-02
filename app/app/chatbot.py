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

def extract_laptop_name(entities: dict) -> Optional[str]:
    """Extract laptop name/model from NER entities"""
    # Combine BRAND tokens
    brand_tokens = entities.get("B-BRAND", []) + entities.get("I-BRAND", [])
    brand = " ".join(brand_tokens).replace("##", "").strip()
    
    # For laptop name queries, we might also check other entity types
    # You can expand this to look for MODEL entities if you add that label
    return brand if brand else None

def extract_params(text: str, entities: dict):
    """Extract recommendation parameters from text and entities"""
    params = {}
    
    # Extract BRAND
    if "B-BRAND" in entities or "I-BRAND" in entities:
        brand_tokens = entities.get("B-BRAND", []) + entities.get("I-BRAND", [])
        brand = "".join(brand_tokens).replace("##", "")
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
    if intent == "goodbye":
        return("Terima kasih! Semoga harimu menyenangkan. 👋")

    elif intent == "greeting":
        return("Halo! Ada yang bisa saya bantu? Mau cari laptop jenis apa?")

    elif intent == "fallback":
        return("Maaf, saya kurang mengerti. Bisa jelaskan lebih detail spesifikasi laptop yang dicari?")

    elif intent == "ask_recommendation":
        # Extract entities and parameters
        entities = predict_entities(user_text)
        params = extract_params(user_text, entities)
        
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
        laptop_name = extract_laptop_name(entities)
        
        # Fallback to regex if NER doesn't find anything
        if not laptop_name:
            match = re.search(r'(?:spesifikasi|spek)\s+([\w\s]+)|([\w\s]+?)\s+(?:gimana|speknya|itu)', user_text, re.IGNORECASE)
            if match:
                laptop_name = match.group(1) if match.group(1) else match.group(2)
        
        if laptop_name and len(laptop_name.strip()) > 2:
            # Search in DataFrame (check Brand or Model columns)
            filtered_laptop = df[
                df['Brand'].str.contains(laptop_name.strip(), case=False, na=False) |
                df['Model'].str.contains(laptop_name.strip(), case=False, na=False)
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
                return f"Maaf, saya tidak menemukan laptop bernama '{laptop_name}'."
        else:
            return "Laptop merk apa yang mau dicek spesifikasinya? (Contoh: 'Spesifikasi Asus TUF')"

    # --- PERBAIKAN LOGIKA ASK_PRICE (with NER) ---
    elif intent == "ask_price":
        # Use NER to extract laptop brand/name
        entities = predict_entities(user_text)
        laptop_name = extract_laptop_name(entities)
        
        # Fallback to regex if NER doesn't find anything
        if not laptop_name:
            match = re.search(r'(?:harga)\s+([\w\s]+)|([\w\s]+?)\s+(?:berapa|harganya)', user_text, re.IGNORECASE)
            if match:
                laptop_name = match.group(1) if match.group(1) else match.group(2)
        
        if laptop_name and len(laptop_name.strip()) > 2:
            filtered_laptop = df[
                df['Brand'].str.contains(laptop_name.strip(), case=False, na=False) |
                df['Model'].str.contains(laptop_name.strip(), case=False, na=False)
            ]
            
            if not filtered_laptop.empty:
                row = filtered_laptop.iloc[0]
                harga = format_idr(row['Final Price'])
                return f"Harga {row['Brand']} {row['Model']} sekitar {harga}."
            else:
                return f"Maaf, harga laptop '{laptop_name}' tidak ditemukan."
        else:
            return "Mau cek harga laptop apa? (Contoh: 'Harga Lenovo Legion berapa?')"

    elif intent == "compare_laptops":
        # Use NER to extract laptop brands
        entities = predict_entities(user_text)
        laptop_name = extract_laptop_name(entities)
        
        # Check if user wants to compare with previous results
        if conversation_memory["last_results"] is not None and len(conversation_memory["last_results"]) >= 2:
            if not laptop_name:  # User says "bandingkan" without specifying
                # Compare top 2 from last results
                df_results = conversation_memory["last_results"]
                laptop1 = df_results.iloc[0]
                laptop2 = df_results.iloc[1]
                
                comparison = f"Perbandingan:\n\n"
                comparison += f"📱 {laptop1['Brand']} {laptop1['Model']}:\n"
                comparison += f"   CPU: {laptop1['CPU']} | RAM: {laptop1['RAM']}GB\n"
                comparison += f"   GPU: {laptop1['GPU']} | Storage: {laptop1['Storage']}GB\n"
                comparison += f"   Screen: {laptop1['Screen']}\" | Harga: {format_idr(laptop1['Final Price'])}\n\n"
                
                comparison += f"📱 {laptop2['Brand']} {laptop2['Model']}:\n"
                comparison += f"   CPU: {laptop2['CPU']} | RAM: {laptop2['RAM']}GB\n"
                comparison += f"   GPU: {laptop2['GPU']} | Storage: {laptop2['Storage']}GB\n"
                comparison += f"   Screen: {laptop2['Screen']}\" | Harga: {format_idr(laptop2['Final Price'])}"
                
                return comparison
        
        # Otherwise ask for specific laptops
        return "Untuk membandingkan, mohon sebutkan dua nama laptop yang ingin Anda bandingkan secara spesifik (misal: 'bandingkan Asus TUF dan Acer Nitro')."
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
    df["Final Price (IDR)"] = format_idr(df["final price"])

    # Drop USD column for user-facing output
    df = df.drop(columns=["final price"])

    return df