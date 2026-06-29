import os
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_available = False

if GEMINI_API_KEY:
    try:
        # pyrefly: ignore [missing-import]
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_available = True
        logger.info("Gemini API key found and configured.")
    except Exception as e:
        logger.error(f"Error configuring Gemini SDK: {e}")
else:
    logger.warning("GEMINI_API_KEY not found. Running receptionist agent in simulation/mock mode.")

# Master Agent Prompt
SYSTEM_INSTRUCTION = """
You are an AI Call Receptionist.
You automatically answer calls when human representatives are busy.
You must communicate entirely through speech. You are not a chatbot. You are a live AI voice agent.
Your purpose is to collect customer details and arrange a callback request.
You must NOT solve customer issues. You must only collect information and notify the appropriate representative.

AGENT BEHAVIOR:
When a call starts, the greeting is:
"Hello. Thank you for calling. All our representatives are currently assisting other customers. I can collect your details and arrange a callback."

INFORMATION COLLECTION:
Collect these details step-by-step:
1. Customer Name: Ask "May I know your name please?"
2. Purpose: Ask "What is the reason for your call today?" (Examples: Booking, Complaint, Support, Technical Issue, Product Inquiry, Service Inquiry, Sales, Refund, Cancellation)
3. Urgency: Ask "Would you consider this urgent?" (Classify into: HIGH, MEDIUM, LOW)
4. Callback Time: Ask "When would you like our representative to contact you?"
5. Contact Number: Ask "Please provide your preferred contact number."

CONFIRMATION:
Once all 5 details are gathered, read back:
"Thank you. Let me confirm." followed by a summary.
Example: "Your name is Arun Kumar. You are calling regarding room booking. The request priority is medium. You prefer a callback tomorrow morning."
Then ask: "Is everything correct?"
If corrections are provided: update the details in memory.

FINAL RESPONSE (when user confirms everything is correct):
Say: "Thank you. I have noted your details. I will inform the appropriate representative. Your request has been recorded and someone from our team will contact you at your preferred time. Thank you for calling."
Then set `call_ended` to true.

RULES:
1. Do not use forms. Do not ask multiple questions together.
2. Keep spoken responses under 15 words whenever possible (except for greeting and final summaries).
3. Do not solve customer issues or talk about troubleshoot steps. Redirect to callback.
4. Language handling: Automatically detect English, Tamil (தமிழ்), or Tanglish (Tamil in Latin letters). Respond in the customer's language/script.
5. If the customer interrupts or speaks, listen completely and continue naturally. Never repeat previous questions or restart the conversation.
6. SLOT CLASSIFICATION: Differentiate strictly between Phone and Callback Time:
   - The `phone` field must contain ONLY contact numbers (digits, +, or spaces).
   - The `callback_time` field must contain ONLY time/date descriptors (e.g. "tomorrow morning", "3 PM", "after 5 PM", "Monday"). Never place contact numbers in the `callback_time` field.

OUTPUT FORMAT:
You MUST respond ONLY with a JSON object matching this schema (do not wrap in markdown ```json blocks):
{
  "detected_language": "english" | "tamil" | "tanglish",
  "context": {
    "name": "extracted name or empty string if not collected yet",
    "phone": "extracted phone or empty string if not collected yet",
    "purpose": "extracted purpose or empty string if not collected yet",
    "priority": "HIGH/MEDIUM/LOW or empty string if not collected yet",
    "callback_time": "extracted callback time or empty string if not collected yet"
  },
  "response_text": "your spoken voice response (under 15 words, except summaries)",
  "is_confirmed": true/false (set to true ONLY if all info is collected AND user confirms it is correct),
  "call_ended": true/false (set to true after final response)
}
"""

class ReceptionistAgent:
    def __init__(self):
        self.is_mock = not gemini_available
        if not self.is_mock:
            try:
                # pyrefly: ignore [missing-import]
                import google.generativeai as genai
                # Setup model
                self.model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={
                        "response_mime_type": "application/json"
                    }
                )
            except Exception as e:
                logger.error(f"Failed to load Gemini model, falling back to mock: {e}")
                self.is_mock = True

    def is_phone_number_value(self, val):
        if not val:
            return False
        cleaned = "".join([c for c in val if c.isdigit()])
        if len(cleaned) < 3:
            return False
            
        time_keywords = ["am", "pm", "morning", "afternoon", "evening", "night", "day", "tomorrow", "today", 
                         "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                         "clock", "o'clock", "hour", "min", "time", "week", "month", "year",
                         "மணி", "காலை", "மாலை", "மதியம்", "இரவு", "நாளை", "இன்று", "நேரம்",
                         "mani", "kaalai", "maalai", "mathiyam", "iravu", "naalai", "inniku", "neram",
                         "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        has_time_word = any(word in val.lower() for word in time_keywords)
        
        # If it contains only digits/spaces/dashes/parentheses/pluses, it's a phone number,
        # unless it is a small number <= 24 (likely hour) or year between 2020 and 2035.
        is_pure_number_pattern = all(c.isdigit() or c in " +-()" for c in val.strip())
        if is_pure_number_pattern:
            try:
                num = int(cleaned)
                if num <= 24 or 2020 <= num <= 2035:
                    return False
            except ValueError:
                pass
            return True
            
        return len(cleaned) >= 5 and not has_time_word

    def is_callback_time_value(self, val):
        if not val:
            return False
            
        # If it looks like a phone number, it's definitely not a callback time
        if self.is_phone_number_value(val):
            return False
            
        time_keywords = ["am", "pm", "morning", "afternoon", "evening", "night", "day", "tomorrow", "today", 
                         "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                         "clock", "o'clock", "hour", "min", "time", "week", "month", "year",
                         "மணி", "காலை", "மாலை", "மதியம்", "இரவு", "நாளை", "இன்று", "நேரம்",
                         "mani", "kaalai", "maalai", "mathiyam", "iravu", "naalai", "inniku", "neram",
                         "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        
        purpose_keywords = ["moto", "iphone", "samsung", "pixel", "nokia", "android", "phone query", "device", "booking", "room", "complaint", "support", "ticket"]
        
        val_lower = val.lower()
        has_time_word = any(word in val_lower for word in time_keywords)
        has_purpose_word = any(word in val_lower for word in purpose_keywords)
        
        if has_purpose_word and not has_time_word:
            return False
            
        digits = "".join([c for c in val if c.isdigit()])
        if digits:
            try:
                num = int(digits)
                if num <= 24:
                    return True
            except ValueError:
                pass
        return len(digits) < 5 or has_time_word

    def clean_and_verify_slots(self, context, user_input=""):
        """
        Validates and swaps slot values if they got mixed up.
        """
        phone = context.get("phone", "")
        if phone is None:
            phone = ""
        phone = str(phone).strip()
        
        callback = context.get("callback_time", "")
        if callback is None:
            callback = ""
        callback = str(callback).strip()
        
        # Check if callback_time is a phone number
        if self.is_phone_number_value(callback):
            context["phone"] = callback
            context["callback_time"] = ""
            
        # Check if callback_time is actually a purpose/product query
        elif callback and not self.is_callback_time_value(callback):
            if context.get("purpose"):
                if callback.lower() not in context["purpose"].lower():
                    context["purpose"] = f"{context['purpose']} - {callback}"
            else:
                context["purpose"] = callback
            context["callback_time"] = ""
            
        # Check if phone is a callback time
        if self.is_callback_time_value(phone) and not self.is_phone_number_value(phone):
            if not callback:
                context["callback_time"] = phone
            context["phone"] = ""

        # Normalize priority based on explicit user input override
        if user_input:
            text = user_input.strip().lower()
            
            is_negated_urgent = False
            import re
            
            # English negations before 'urgent'
            if "urgent" in text:
                for neg in ["not", "no", "don't", "dont", "non"]:
                    if re.search(rf"\b{neg}\b", text):
                        neg_idx = text.find(neg)
                        urg_idx = text.find("urgent")
                        if neg_idx < urg_idx:
                            is_negated_urgent = True
                            break
                            
            # Tamil/Tanglish negations after 'avsaram' / 'அவசரம்'
            if not is_negated_urgent:
                for urg in ["avsaram", "அவசரம்"]:
                    if urg in text:
                        for neg in ["illa", "illai", "இல்லை"]:
                            if re.search(rf"\b{neg}\b", text):
                                urg_idx = text.find(urg)
                                neg_idx = text.find(neg)
                                if urg_idx < neg_idx:
                                    is_negated_urgent = True
                                    break

            if is_negated_urgent:
                context["priority"] = "MEDIUM"
            elif "yes urgent" in text or "urgent" in text:
                context["priority"] = "HIGH"

        return context

    def get_summary_text(self, context, r, detected_lang):
        summary_intro = r["confirm_intro"]
        if detected_lang == "english":
            summary_details = f"Your name is {context['name']}. You are calling regarding {context['purpose']}. The request priority is {context['priority'].lower()}. You prefer a callback {context['callback_time']}. Contact number is {context['phone']}."
        elif detected_lang == "tamil":
            summary_details = f"உங்கள் பெயர் {context['name']}. அழைப்பின் காரணம் {context['purpose']}. முன்னுரிமை {context['priority']}. நீங்கள் {context['callback_time']} தொடர்பு கொள்ள விரும்புகிறீர்கள். தொடர்பு எண் {context['phone']}."
        else:
            summary_details = f"Unga name {context['name']}. Call reason {context['purpose']}. Priority {context['priority']}. Callback time {context['callback_time']}. Contact number {context['phone']}."
            
        return f"{summary_intro} {summary_details} {r['ask_correct']}"

    def process_turn(self, conversation_history, user_input, current_context, caller_phone=""):
        """
        Processes a turn of the conversation.
        conversation_history: List of dicts representing the chat: [{"role": "user"/"model", "parts": ["text"]}]
        user_input: The latest transcribed speech from the user
        current_context: The backend in-memory dictionary call_context
        caller_phone: Default phone number of caller (e.g. from Twilio caller ID)
        """
        if self.is_mock:
            return self._mock_process_turn(user_input, current_context, caller_phone)

        try:
            # pyrefly: ignore [missing-import]
            import google.generativeai as genai
            
            # Format history for the Gemini API
            # Strict alternating user -> model format starting with user message
            contents = []
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["parts"][0]]})
            
            # Note: We do NOT append user_input here because it is already appended to conversation_history by main.py
            
            response = self.model.generate_content(contents)
            text = response.text.strip()
            
            # Clean Markdown formatting ticks if present
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
                
            response_json = json.loads(text)
            
            # Clean up the output context if needed
            if "context" not in response_json:
                response_json["context"] = {}
            for k in ["name", "phone", "purpose", "priority", "callback_time"]:
                if k not in response_json["context"]:
                    response_json["context"][k] = current_context.get(k, "")
            
            # Run slot verification & cleanup
            response_json["context"] = self.clean_and_verify_slots(response_json["context"], user_input)
            
            return response_json
            
        except Exception as e:
            logger.error(f"Gemini API Error: {e}. Falling back to smart mock agent.")
            return self._mock_process_turn(user_input, current_context, caller_phone)

    def _mock_process_turn(self, user_input, current_context, caller_phone=""):
        """
        A smart rule-based parser that handles slots flexibly and detects Tamil/Tanglish.
        Ensures the conversational flow continues naturally even in demo/fallback mode.
        """
        text = user_input.strip().lower()
        context = current_context.copy()
        
        # Multilingual Keyword Detection
        detected_lang = "english"
        tamil_keywords = ["வணக்கம்", "பேர்", "நன்றி", "ஆமாம்", "இல்லை", "உதவி", "அவசரம்", "விபரம்", "தேவை"]
        tanglish_keywords = ["vanakkam", "peru", "enoda", "aama", "illai", "avsaram", "priority", "timing", "call panunga", "enakku"]
        
        if any(kw in text for kw in tamil_keywords):
            detected_lang = "tamil"
        elif any(kw in text for kw in tanglish_keywords):
            detected_lang = "tanglish"
            
        r_maps = {
            "english": {
                "ask_name": "May I know your name please?",
                "ask_purpose": "What is the reason for your call today?",
                "ask_urgency": "Would you consider this urgent?",
                "ask_callback": "When would you like our representative to contact you?",
                "ask_phone": f"Is {caller_phone or 'this'} the best number to reach you?",
                "ask_custom_phone": "Please provide your preferred contact number.",
                "confirm_intro": "Thank you. Let me confirm.",
                "ask_correct": "Is everything correct?",
                "final_bye": "Thank you. I have noted your details. I will inform the appropriate representative. Your request has been recorded and someone from our team will contact you at your preferred time. Thank you for calling."
            },
            "tamil": {
                "ask_name": "உங்கள் பெயர் என்னவென்று தெரிந்து கொள்ளலாமா?",
                "ask_purpose": "நீங்கள் எதற்காக அழைக்கிறீர்கள்?",
                "ask_urgency": "இது அவசரமான அழைப்பா?",
                "ask_callback": "எங்கள் பிரதிநிதி உங்களை எப்போது தொடர்பு கொள்ள வேண்டும்?",
                "ask_phone": f"உங்களை தொடர்பு கொள்ள {caller_phone or 'இந்த'} எண் சிறந்ததா?",
                "ask_custom_phone": "உங்களுக்கு விருப்பமான தொடர்பு எண்ணை வழங்கவும்.",
                "confirm_intro": "நன்றி. நான் உறுதிப்படுத்துகிறேன்.",
                "ask_correct": "எல்லாம் சரியாக இருக்கிறதா?",
                "final_bye": "நன்றி. உங்கள் விவரங்களை நான் குறித்துக்கொண்டேன். தகுந்த பிரதிநிதியிடம் தெரிவிப்பேன். உங்களது விருப்பமான நேரத்தில் தொடர்பு கொள்ளப்படும். அழைத்ததற்கு நன்றி."
            },
            "tanglish": {
                "ask_name": "Unga name enna nu therinjukalama?",
                "ask_purpose": "Call panna reason enna nu solla mudiyuma?",
                "ask_urgency": "Ithu romba urgent ah?",
                "ask_callback": "Enga team eppo ungaluku call back pannanum?",
                "ask_phone": f"Ungalai reach panna {caller_phone or 'inthathu'} correct number ah?",
                "ask_custom_phone": "Unga preferred contact number ah solunga.",
                "confirm_intro": "Rumba nandri. Oru vaati confirm pannikalam.",
                "ask_correct": "Ellam correct ah iruka?",
                "final_bye": "Nandri. details note panniyachu. representative eppo contact pannuvanga. call pannathuku nandri."
            }
        }
        
        r = r_maps[detected_lang]
        response_text = ""
        is_confirmed = False
        call_ended = False

        # If name is not collected yet
        if not context.get("name"):
            # If user just said hello/hi without name details
            if text in ["hello", "hi", "hey", "vanakkam", "வணக்கம்", "hola", "call"]:
                response_text = r["ask_name"]
            else:
                # Extract name (strip greeting prefixes and phrases)
                cleaned_name = user_input
                for prefix in ["my name is", "i am", "en peru", "peru", "en peyar", "this is", "iam", "per"]:
                    if prefix in cleaned_name.lower():
                        idx = cleaned_name.lower().find(prefix) + len(prefix)
                        cleaned_name = cleaned_name[idx:]
                cleaned_name = cleaned_name.replace(",", "").replace(".", "").strip()
                
                context["name"] = cleaned_name.title() if cleaned_name else "Valued Caller"
                response_text = r["ask_purpose"]

        # If purpose is not collected yet
        elif not context.get("purpose"):
            context["purpose"] = user_input.capitalize()
            response_text = r["ask_urgency"]

        # If urgency is not collected yet
        elif not context.get("priority"):
            # Check for high urgency triggers
            urgency_triggers = ["yes", "urgent", "avsaram", "aama", "high", "crying", "broken", "emergency", "immediate", "rumba", "ஆமாம்", "அவசரம்"]
            no_triggers = ["no", "medium", "normal", "illa", "low", "இல்லை", "medium priority"]
            
            is_negated_urgent = False
            import re
            
            # English negations before 'urgent'
            if "urgent" in text:
                for neg in ["not", "no", "don't", "dont", "non"]:
                    if re.search(rf"\b{neg}\b", text):
                        neg_idx = text.find(neg)
                        urg_idx = text.find("urgent")
                        if neg_idx < urg_idx:
                            is_negated_urgent = True
                            break
                            
            # Tamil/Tanglish negations after 'avsaram' / 'அவசரம்'
            if not is_negated_urgent:
                for urg in ["avsaram", "அவசரம்"]:
                    if urg in text:
                        for neg in ["illa", "illai", "இல்லை"]:
                            if re.search(rf"\b{neg}\b", text):
                                urg_idx = text.find(urg)
                                neg_idx = text.find(neg)
                                if urg_idx < neg_idx:
                                    is_negated_urgent = True
                                    break

            if is_negated_urgent:
                context["priority"] = "MEDIUM"
            elif any(w in text for w in urgency_triggers):
                context["priority"] = "HIGH"
            elif any(w in text for w in no_triggers):
                context["priority"] = "LOW"
            else:
                context["priority"] = "MEDIUM"
                
            response_text = r["ask_callback"]

        # If callback time is not collected yet
        elif not context.get("callback_time"):
            if self.is_phone_number_value(user_input):
                context["phone"] = user_input
                response_text = r["ask_callback"]
            elif not self.is_callback_time_value(user_input):
                # The user input is not a callback time value! It's a purpose/product query.
                if context.get("purpose"):
                    if user_input.lower() not in context["purpose"].lower():
                        context["purpose"] = f"{context['purpose']} - {user_input}"
                else:
                    context["purpose"] = user_input
                response_text = r["ask_callback"]
            else:
                context["callback_time"] = user_input
                # If phone is already collected (e.g. stated early), go straight to confirmation summary
                if context.get("phone") and context["phone"] != "PENDING":
                    response_text = self.get_summary_text(context, r, detected_lang)
                else:
                    response_text = r["ask_custom_phone"]
                    context["phone"] = "PENDING"

        # If contact number is not collected yet
        elif not context.get("phone") or context["phone"] == "PENDING":
            # Extract digits
            digits = "".join([c for c in user_input if c.isdigit() or c == "+"])
            context["phone"] = digits if len(digits) >= 7 else user_input
            
            response_text = self.get_summary_text(context, r, detected_lang)

        # If confirmation is pending
        elif not is_confirmed:
            yes_triggers = ["yes", "correct", "aama", "am", "right", "sari", "ok", "ஆமாம்", "sariyo", "correct ah iruku"]
            if any(w in text for w in yes_triggers):
                is_confirmed = True
                response_text = r["final_bye"]
                call_ended = True
            else:
                # Correction trigger: restart callback selection
                context["callback_time"] = ""
                response_text = "Let's correct that. " + r["ask_callback"]
        else:
            response_text = r["final_bye"]
            call_ended = True
            
        return {
            "detected_language": detected_lang,
            "context": self.clean_and_verify_slots(context, user_input),
            "response_text": response_text,
            "is_confirmed": is_confirmed,
            "call_ended": call_ended
        }
