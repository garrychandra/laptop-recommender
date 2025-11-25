from app.recommender import load_data, recommend
from app.nlp_pipeline import predict_intent, predict_entities
import re

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

    if intent == "greeting":
        return "Halo! Apa kebutuhan laptop Anda?"

    if intent == "goodbye":
        return "Terima kasih! Sampai jumpa 👋"

    if intent == "ask_recommendation":
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

    return "Maaf, saya belum mengerti maksud Anda."


def format_results(df):
    df = df.copy()
    # Convert USD → IDR for display
    df["Final Price (IDR)"] = (df["final price"] * 16000).astype(int)

    # Drop USD column for user-facing output
    df = df.drop(columns=["final price"])

    return df