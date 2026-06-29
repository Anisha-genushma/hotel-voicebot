import re

# Hardcoded hotel FAQ dataset containing 50 entries
FAQ_DATA = [
    {
        "question": "Is any room available?",
        "answer": "Yes sir, Deluxe Rooms and Executive Rooms are currently available."
    },
    {
        "question": "How many rooms are available now?",
        "answer": "Currently 12 rooms are available for booking."
    },
    {
        "question": "Is this hotel couple friendly?",
        "answer": "Yes, this hotel is couple friendly with valid government ID proof."
    },
    {
        "question": "Do you have AC rooms?",
        "answer": "Yes, all rooms are fully air-conditioned."
    },
    {
        "question": "Do you have WiFi?",
        "answer": "Yes, free high-speed WiFi is available throughout the hotel."
    },
    {
        "question": "Is parking available?",
        "answer": "Yes, free parking is available for all guests."
    },
    {
        "question": "What is the check-in time?",
        "answer": "Check-in starts at 12 PM."
    },
    {
        "question": "What is the check-out time?",
        "answer": "Check-out time is 11 AM."
    },
    {
        "question": "Do you provide breakfast?",
        "answer": "Yes, complimentary breakfast is included."
    },
    {
        "question": "Do you have family rooms?",
        "answer": "Yes, family rooms are available."
    },
    {
        "question": "Is early check-in possible?",
        "answer": "Early check-in depends on room availability."
    },
    {
        "question": "Can I cancel my booking?",
        "answer": "Yes, cancellation is available according to hotel policy."
    },
    {
        "question": "Do you allow pets?",
        "answer": "Sorry, pets are not allowed."
    },
    {
        "question": "Is hot water available?",
        "answer": "Yes, 24-hour hot water is available."
    },
    {
        "question": "Do you have room service?",
        "answer": "Yes, room service is available 24 hours."
    },
    {
        "question": "Do you have a restaurant?",
        "answer": "Yes, the hotel has a multi-cuisine restaurant."
    },
    {
        "question": "Do you provide airport pickup?",
        "answer": "Yes, airport pickup is available on request."
    },
    {
        "question": "Is smoking allowed?",
        "answer": "Smoking is allowed only in designated areas."
    },
    {
        "question": "Do you have a swimming pool?",
        "answer": "Yes, guests can access the swimming pool."
    },
    {
        "question": "Do you have a gym?",
        "answer": "Yes, the fitness center is open from 6 AM to 10 PM."
    },
    {
        "question": "Can I pay by card?",
        "answer": "Yes, we accept all major debit and credit cards."
    },
    {
        "question": "Do you accept UPI?",
        "answer": "Yes, UPI payments are accepted."
    },
    {
        "question": "Is there a lift?",
        "answer": "Yes, lift facilities are available."
    },
    {
        "question": "Do you provide extra beds?",
        "answer": "Extra beds are available with additional charges."
    },
    {
        "question": "Is laundry service available?",
        "answer": "Yes, laundry service is available."
    },
    {
        "question": "Do you have conference halls?",
        "answer": "Yes, conference halls are available."
    },
    {
        "question": "Is the hotel safe?",
        "answer": "Yes, CCTV and 24-hour security are available."
    },
    {
        "question": "Do you have CCTV?",
        "answer": "Yes, CCTV surveillance is active throughout the hotel."
    },
    {
        "question": "Can unmarried couples stay?",
        "answer": "Yes, unmarried couples are welcome with valid ID proof."
    },
    {
        "question": "Are local IDs accepted?",
        "answer": "Yes, local IDs are accepted as per policy."
    },
    {
        "question": "How much is the Deluxe Room?",
        "answer": "The Deluxe Room starts from ₹2499 per night."
    },
    {
        "question": "How much is the Suite Room?",
        "answer": "The Suite Room starts from ₹4999 per night."
    },
    {
        "question": "Do you provide towels?",
        "answer": "Yes, complimentary towels are provided."
    },
    {
        "question": "Do rooms have TV?",
        "answer": "Yes, all rooms include Smart TVs."
    },
    {
        "question": "Do rooms have AC?",
        "answer": "Yes, every room has air conditioning."
    },
    {
        "question": "Do rooms have a balcony?",
        "answer": "Selected rooms have balconies."
    },
    {
        "question": "Is drinking water provided?",
        "answer": "Yes, complimentary bottled water is provided."
    },
    {
        "question": "Can I extend my stay?",
        "answer": "Yes, subject to room availability."
    },
    {
        "question": "Do you have 24-hour reception?",
        "answer": "Yes, the reception is open 24 hours."
    },
    {
        "question": "Is luggage storage available?",
        "answer": "Yes, luggage storage is available."
    },
    {
        "question": "Can I book online?",
        "answer": "Yes, online booking is available."
    },
    {
        "question": "Do you provide invoices?",
        "answer": "Yes, GST invoices are provided."
    },
    {
        "question": "Is there a power backup?",
        "answer": "Yes, generator backup is available."
    },
    {
        "question": "Is WiFi free?",
        "answer": "Yes, WiFi is completely free."
    },
    {
        "question": "Is breakfast included?",
        "answer": "Yes, breakfast is included in selected plans."
    },
    {
        "question": "Do you have meeting rooms?",
        "answer": "Yes, meeting rooms are available."
    },
    {
        "question": "Do you have spa facilities?",
        "answer": "Yes, spa services are available."
    },
    {
        "question": "Do you have children's play area?",
        "answer": "Yes, a kids play area is available."
    },
    {
        "question": "Do you provide taxi service?",
        "answer": "Yes, taxi booking assistance is available."
    },
    {
        "question": "Can I bring guests?",
        "answer": "Visitors are allowed according to hotel policy."
    }
]

# Normalization mapping to resolve synonyms into standardized root tokens
NORMALIZE_MAP = {
    # Availability synonyms mapping to 'available'
    "free": "available",
    "vacancy": "available",
    "vacancies": "available",
    "vacant": "available",
    "empty": "available",
    "full": "available",
    
    # Room synonyms mapping to 'room'
    "rooms": "room",
    "accommodation": "room",
    
    # Count synonyms mapping to 'count'
    "left": "count",
    "remaining": "count",
    "number": "count",
    "numbers": "count",
    "many": "count",
    
    # Couple friendly mapping to 'couple'
    "couples": "couple",
    "unmarried": "couple",
    "partners": "couple",
    "entry": "couple",
    
    # Pricing mapping to 'price'
    "cost": "price",
    "charge": "price",
    "charges": "price",
    "rate": "price",
    "tariff": "price",
    "fees": "price",
    "fee": "price",
    "how much": "price",
    
    # Payments
    "payment": "pay",
    "card": "pay",
    "credit": "pay",
    "debit": "pay",
    "upi": "pay",
    "gpay": "pay",
    "phonepe": "pay",
    "paytm": "pay",
    
    # Amenities mapping to standard terms
    "internet": "wifi",
    "broadband": "wifi",
    "net": "wifi",
    "car": "parking",
    "vehicle": "parking",
    "garage": "parking",
    "park": "parking",
    "fitness": "gym",
    "workout": "gym",
    "exercise": "gym",
    "pool": "swimming",
    "swim": "swimming",
    "geyser": "hotwater",
    "shower": "hotwater",
    "gyser": "hotwater",
    "smoke": "smoking",
    "cigarette": "smoking",
    "pet": "pets",
    "dog": "pets",
    "cat": "pets",
    "animals": "pets",
    "pickup": "taxi",
    "drop": "taxi",
    "cab": "taxi",
    "shuttle": "taxi",
    "transfer": "taxi",
    "meals": "restaurant",
    "meal": "restaurant",
    "food": "restaurant",
    "dining": "restaurant",
    "dine": "restaurant",
    "eat": "restaurant",
    "safe": "security",
    "safety": "security",
    "cameras": "security",
    "camera": "security",
    "cctv": "security",
}

# Stop words to filter out for better semantic overlap matching
# Filter out common auxiliary verbs, pronouns, and overly generic hotel query verbs
STOP_WORDS = {
    "is", "a", "an", "the", "do", "you", "have", "for", "to", "in", "of", 
    "at", "with", "this", "can", "i", "we", "they", "he", "she", "it", 
    "are", "on", "any", "now", "there", "what", "who", "where", "how",
    "allowed", "possible", "provide", "provides", "accept", "accepted",
    "take", "booking", "book", "get", "stay", "staying", "go", "will", "would",
    "please", "sir", "mam", "hotel", "friendly", "friend"
}

def clean_text(text):
    """
    Cleans punctuation and lowercases the text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Replace punctuation with spaces
    return text

def get_tokens(text):
    """
    Splits text into cleaned tokens and removes common stop words, then maps synonyms to root words.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    normalized = []
    for t in tokens:
        if t in STOP_WORDS:
            continue
        # Map synonyms to standardized root terms
        if t in NORMALIZE_MAP:
            normalized.append(NORMALIZE_MAP[t])
        else:
            normalized.append(t)
    return set(normalized)

def get_similarity(query_tokens, question_tokens):
    """
    Computes Jaccard Similarity on the normalized token sets.
    """
    intersection = query_tokens.intersection(question_tokens)
    union = query_tokens.union(question_tokens)
    
    if not union:
        return 0.0
    return len(intersection) / len(union)

def find_best_match(user_query):
    """
    Finds the best matching question in FAQ_DATA based on Jaccard semantic similarity of normalized tokens.
    Returns the matching answer string, or None if match score is below threshold.
    """
    query_tokens = get_tokens(user_query)
    if not query_tokens:
        return None
        
    best_match = None
    max_score = 0.0
    
    for faq in FAQ_DATA:
        q_tokens = get_tokens(faq["question"])
        score = get_similarity(query_tokens, q_tokens)
        if score > max_score:
            max_score = score
            best_match = faq
            
    # Score threshold: 0.15 is ideal for matching normalized keyword groups
    if max_score >= 0.15:
        return best_match["answer"]
    return None
