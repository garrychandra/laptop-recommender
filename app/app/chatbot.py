from app.recommender import load_data, recommend
from app.nlp_pipeline import predict_intent, predict_entities
import re
import string
import sys

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
        
        results = recommend(df, **params)
        results = format_results(results)
        return "Berikut rekomendasi laptop:\n" + results.to_string(index=False)

    # --- PERBAIKAN LOGIKA ASK_SPECS ---
    elif intent == "ask_specs":
        # Regex diperketat: Wajib menangkap kata setelah 'spesifikasi' atau sebelum 'gimana'
        match = re.search(r'(?:spesifikasi|spek)\s+([\w\s]+)|([\w\s]+?)\s+(?:gimana|speknya|itu)', raw_user_input, re.IGNORECASE)

        laptop_name = None
        if match:
            # Ambil group yang tidak None
            laptop_name = match.group(1) if match.group(1) else match.group(2)

        # Cek validitas nama laptop (minimal 3 huruf agar tidak asal tebak)
        if laptop_name and len(laptop_name.strip()) > 2:
            filtered_laptop = df[df['Laptop'].str.contains(laptop_name.strip(), case=False, na=False)]

            if not filtered_laptop.empty:
                row = filtered_laptop.iloc[0]
                return(f"Spesifikasi {row['Brand']} {row['Model']}:")
                return(f"   • CPU: {row['CPU']}")
                return(f"   • RAM: {row['RAM']}GB | Storage: {row['Storage']}GB")
                return(f"   • GPU: {row['GPU']}")
            else:
                return(f"Maaf, saya tidak menemukan laptop bernama '{laptop_name}'.")
        else:
            # Jika intent terdeteksi ask_specs tapi tidak ada nama laptop
            return("Laptop merk apa yang mau dicek spesifikasinya? (Contoh: 'Spesifikasi Asus TUF')")

    # --- PERBAIKAN LOGIKA ASK_PRICE ---
    elif intent == "ask_price":
        match = re.search(r'(?:harga)\s+([\w\s]+)|([\w\s]+?)\s+(?:berapa|harganya)', raw_user_input, re.IGNORECASE)

        laptop_name = None
        if match:
            laptop_name = match.group(1) if match.group(1) else match.group(2)

        if laptop_name and len(laptop_name.strip()) > 2:
            filtered_laptop = df[df['Laptop'].str.contains(laptop_name.strip(), case=False, na=False)]
            if not filtered_laptop.empty:
                row = filtered_laptop.iloc[0]
                harga = format_idr(row['Final Price'])
                return(f"Harga {row['Brand']} {row['Model']} sekitar {harga}.")
            else:
                return(f"Maaf, harga laptop '{laptop_name}' tidak ditemukan.")
        else:
            return("Mau cek harga laptop apa? (Contoh: 'Harga Lenovo Legion berapa?')")

    elif intent == "compare_laptops":
        # Simple logic for comparing laptops (can be expanded)
        return("Untuk membandingkan, mohon sebutkan dua nama laptop yang ingin Anda bandingkan secara spesifik (misal: 'bandingkan Asus TUF dan Acer Nitro').")
    elif intent == "clarify_requirement":
        # Simple logic for clarifying requirements
        return("Baik, saya mengerti. Bisakah Anda memberikan lebih banyak detail tentang kebutuhan Anda? Misalnya, untuk keperluan apa laptop ini akan digunakan (gaming, kerja, desain, dll.), atau fitur spesifik lainnya?")

    else:
        return("Maaf, saya belum paham maksud Anda.")

def format_results(df):
    df = df.copy()
    # Convert USD → IDR for display
    df["Final Price (IDR)"] = format_idr(df["final price"])

    # Drop USD column for user-facing output
    df = df.drop(columns=["final price"])

    return df