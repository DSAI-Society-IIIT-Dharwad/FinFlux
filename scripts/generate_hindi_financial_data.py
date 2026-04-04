"""Generate synthetic Hindi/Hinglish financial conversation data with TTS audio.

Creates realistic financial domain training data covering:
- EMI payments and schedules
- Loan discussions (home, personal, car, education)
- SIP and mutual fund investments
- Insurance discussions
- Fixed deposits and savings
- Credit card payments
- Tax planning
- Budget and expense discussions
- Investment returns and portfolio

Output: WAV files + manifest CSV compatible with training pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import sys
import time
import wave
from pathlib import Path
from typing import Any

# Set Coqui Terms of Service Agreement for non-interactive use
os.environ["COQUI_TOS_AGREED"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Hindi financial conversation templates
# ---------------------------------------------------------------------------

HINDI_TEMPLATES: list[dict[str, str]] = [
    # EMI
    {"text": "मेरी EMI इस महीने पाँच हज़ार रुपये है", "language": "hi"},
    {"text": "कार लोन की EMI बहुत ज़्यादा हो गई है", "language": "hi"},
    {"text": "होम लोन की किस्त समय पर भरनी होगी", "language": "hi"},
    {"text": "EMI की तारीख पंद्रह तारीख है हर महीने", "language": "hi"},
    {"text": "मैं अगले महीने से EMI बढ़ाना चाहता हूँ", "language": "hi"},
    {"text": "बैंक ने EMI बाउंस का नोटिस भेजा है", "language": "hi"},
    {"text": "दस हज़ार रुपये की EMI हर महीने कटती है", "language": "hi"},
    {"text": "EMI में कुछ कमी करवा सकते हैं क्या", "language": "hi"},
    {"text": "तीन साल की EMI बाकी है अभी", "language": "hi"},
    {"text": "पर्सनल लोन की EMI सात हज़ार है", "language": "hi"},
    {"text": "EMI का भुगतान ऑनलाइन करना है", "language": "hi"},
    {"text": "मेरी सारी EMI मिलाकर बीस हज़ार हो जाती है", "language": "hi"},

    # Loans
    {"text": "होम लोन के लिए आवेदन करना है", "language": "hi"},
    {"text": "पर्सनल लोन का ब्याज दर क्या है", "language": "hi"},
    {"text": "एजुकेशन लोन पर सब्सिडी मिलती है", "language": "hi"},
    {"text": "कार लोन पाँच साल के लिए लेना है", "language": "hi"},
    {"text": "लोन का प्रीपेमेंट करने से फायदा होगा", "language": "hi"},
    {"text": "बैंक ने लोन रिजेक्ट कर दिया है", "language": "hi"},
    {"text": "गोल्ड लोन पर ब्याज कम है", "language": "hi"},
    {"text": "बिज़नेस लोन के लिए दस्तावेज़ चाहिए", "language": "hi"},
    {"text": "लोन का बैलेंस चेक करना है", "language": "hi"},
    {"text": "होम लोन पर टैक्स छूट मिलती है", "language": "hi"},
    {"text": "लोन ट्रांसफर करने से ब्याज कम होगा", "language": "hi"},
    {"text": "दो लाख का पर्सनल लोन चाहिए", "language": "hi"},

    # SIP & Mutual Funds
    {"text": "म्यूचुअल फंड में SIP शुरू करनी है", "language": "hi"},
    {"text": "हर महीने पाँच हज़ार की SIP करनी है", "language": "hi"},
    {"text": "SIP बढ़ाकर दस हज़ार करनी है", "language": "hi"},
    {"text": "इक्विटी फंड में निवेश करना सही रहेगा", "language": "hi"},
    {"text": "म्यूचुअल फंड का रिटर्न अच्छा आया है", "language": "hi"},
    {"text": "ELSS फंड में टैक्स सेविंग हो जाएगी", "language": "hi"},
    {"text": "SIP की तारीख बदलनी है", "language": "hi"},
    {"text": "लार्ज कैप फंड में पैसा लगाना है", "language": "hi"},
    {"text": "SIP बंद करनी है अगले महीने से", "language": "hi"},
    {"text": "म्यूचुअल फंड से पैसा निकालना है", "language": "hi"},
    {"text": "इंडेक्स फंड में निवेश करना बेहतर है", "language": "hi"},
    {"text": "SIP में स्टेप अप करने से ज़्यादा रिटर्न मिलेगा", "language": "hi"},

    # Insurance
    {"text": "टर्म इंश्योरेंस करवाना ज़रूरी है", "language": "hi"},
    {"text": "हेल्थ इंश्योरेंस का प्रीमियम कितना है", "language": "hi"},
    {"text": "लाइफ इंश्योरेंस पॉलिसी रिन्यू करनी है", "language": "hi"},
    {"text": "बीमा क्लेम के लिए आवेदन करना है", "language": "hi"},
    {"text": "फैमिली फ्लोटर प्लान लेना चाहिए", "language": "hi"},
    {"text": "इंश्योरेंस का कवरेज बढ़ाना है", "language": "hi"},
    {"text": "मोटर इंश्योरेंस एक्सपायर हो गया है", "language": "hi"},
    {"text": "पचास लाख का टर्म प्लान लेना है", "language": "hi"},

    # FD & Savings
    {"text": "फिक्स्ड डिपॉज़िट में पैसा रखना है", "language": "hi"},
    {"text": "बचत खाते में बैलेंस कम है", "language": "hi"},
    {"text": "RD हर महीने दो हज़ार की है", "language": "hi"},
    {"text": "FD की ब्याज दर कम हो गई है", "language": "hi"},
    {"text": "पीपीएफ में पैसा जमा करना है", "language": "hi"},
    {"text": "सेविंग्स अकाउंट में मिनिमम बैलेंस रखना है", "language": "hi"},
    {"text": "FD तोड़ने पर पेनल्टी लगेगी", "language": "hi"},
    {"text": "सीनियर सिटीज़न FD पर ब्याज ज़्यादा मिलेगा", "language": "hi"},

    # Credit Card
    {"text": "क्रेडिट कार्ड का बिल बहुत ज़्यादा आया है", "language": "hi"},
    {"text": "क्रेडिट कार्ड की लिमिट बढ़ाना है", "language": "hi"},
    {"text": "क्रेडिट कार्ड बिल की ड्यू डेट कल है", "language": "hi"},
    {"text": "मिनिमम ड्यू अमाउंट भरना ज़रूरी है", "language": "hi"},
    {"text": "क्रेडिट कार्ड पर कैशबैक मिला है", "language": "hi"},

    # Tax
    {"text": "इनकम टैक्स रिटर्न भरना है", "language": "hi"},
    {"text": "टैक्स बचाने के लिए निवेश करना है", "language": "hi"},
    {"text": "सेक्शन अस्सी C में छूट मिलेगी", "language": "hi"},
    {"text": "अग्रिम कर का भुगतान करना है", "language": "hi"},
    {"text": "GST रिटर्न फाइल करना है", "language": "hi"},
    {"text": "TDS कटा है सैलरी से", "language": "hi"},

    # Budget & Expenses
    {"text": "इस महीने का बजट बना लो", "language": "hi"},
    {"text": "खर्चे बहुत बढ़ गए हैं इस महीने", "language": "hi"},
    {"text": "बचत करनी होगी अगले महीने से", "language": "hi"},
    {"text": "किराया दस हज़ार देना है", "language": "hi"},
    {"text": "बिजली का बिल तीन हज़ार आया है", "language": "hi"},
    {"text": "ग्रोसरी का खर्चा पाँच हज़ार है महीने का", "language": "hi"},
    {"text": "सैलरी से बचत नहीं हो पा रही", "language": "hi"},
    {"text": "इमरजेंसी फंड बनाना शुरू करो", "language": "hi"},

    # Investments
    {"text": "शेयर बाज़ार में पैसा लगाना है", "language": "hi"},
    {"text": "सोने में निवेश करना सुरक्षित है", "language": "hi"},
    {"text": "प्रॉपर्टी ख़रीदने का सही समय है", "language": "hi"},
    {"text": "डिविडेंड इनकम अच्छी आ रही है", "language": "hi"},
    {"text": "पोर्टफोलियो में डायवर्सिफिकेशन करो", "language": "hi"},
    {"text": "NPS में निवेश करने से टैक्स बचेगा", "language": "hi"},
    {"text": "स्टॉक बेचकर प्रॉफिट बुक करो", "language": "hi"},
    {"text": "रियल एस्टेट में निवेश रिस्की है", "language": "hi"},
]

HINGLISH_TEMPLATES: list[dict[str, str]] = [
    # EMI - Hinglish
    {"text": "EMI manage ho jayegi is month", "language": "hinglish"},
    {"text": "mera car loan ka EMI bahut zyada hai yaar", "language": "hinglish"},
    {"text": "home loan EMI time pe pay karni hogi", "language": "hinglish"},
    {"text": "EMI bounce ho gayi last month bank ne notice bheja", "language": "hinglish"},
    {"text": "next month se EMI increase karni hai", "language": "hinglish"},
    {"text": "personal loan ki EMI seven thousand hai monthly", "language": "hinglish"},
    {"text": "total EMI milake twenty thousand ho jati hai", "language": "hinglish"},
    {"text": "EMI reduction karwa sakte hain kya bank se", "language": "hinglish"},
    {"text": "three years ki EMI baki hai abhi loan pe", "language": "hinglish"},
    {"text": "online EMI payment karna hai aaj", "language": "hinglish"},
    {"text": "education loan ki EMI start hogi six months baad", "language": "hinglish"},
    {"text": "EMI auto debit set karo account se", "language": "hinglish"},

    # Loan - Hinglish
    {"text": "home loan ke liye apply karna hai bank mein", "language": "hinglish"},
    {"text": "personal loan ka interest rate kya hai abhi", "language": "hinglish"},
    {"text": "loan lena safe hai kya is time pe", "language": "hinglish"},
    {"text": "car loan five years ke liye le rahe hain", "language": "hinglish"},
    {"text": "loan prepayment karne se benefit hoga", "language": "hinglish"},
    {"text": "bank ne loan reject kar diya low CIBIL score ki wajah se", "language": "hinglish"},
    {"text": "gold loan pe interest kam hai comparatively", "language": "hinglish"},
    {"text": "business loan ke liye documents chahiye kaunse", "language": "hinglish"},
    {"text": "loan balance check karna hai online", "language": "hinglish"},
    {"text": "home loan pe tax benefit milta hai section mein", "language": "hinglish"},
    {"text": "loan transfer karne se interest rate kam milega", "language": "hinglish"},
    {"text": "do lakh ka personal loan chahiye urgently", "language": "hinglish"},
    {"text": "CIBIL score improve karna padega loan ke liye", "language": "hinglish"},

    # SIP & MF - Hinglish
    {"text": "mutual fund mein SIP start karni hai next month se", "language": "hinglish"},
    {"text": "SIP badha dete hain two thousand se five thousand", "language": "hinglish"},
    {"text": "equity fund mein invest karna sahi rahega long term ke liye", "language": "hinglish"},
    {"text": "mutual fund ka return accha aaya hai is saal", "language": "hinglish"},
    {"text": "ELSS fund mein tax saving ho jayegi", "language": "hinglish"},
    {"text": "SIP ki date change karni hai fifteen se first", "language": "hinglish"},
    {"text": "large cap fund mein paisa lagana hai", "language": "hinglish"},
    {"text": "SIP band karni hai next month se temporarily", "language": "hinglish"},
    {"text": "mutual fund se paisa withdraw karna hai partial", "language": "hinglish"},
    {"text": "index fund mein invest karna better hai long run mein", "language": "hinglish"},
    {"text": "SIP step up karne se zyada returns milenge", "language": "hinglish"},
    {"text": "five thousand monthly SIP shuru karo index fund mein", "language": "hinglish"},
    {"text": "portfolio rebalance karna chahiye quarterly", "language": "hinglish"},

    # Insurance - Hinglish
    {"text": "term insurance karwana zaroori hai family ke liye", "language": "hinglish"},
    {"text": "health insurance ka premium kitna hai yearly", "language": "hinglish"},
    {"text": "life insurance policy renew karni hai next week", "language": "hinglish"},
    {"text": "insurance claim ke liye apply karna hai hospital bills ka", "language": "hinglish"},
    {"text": "family floater plan le lo sab ke liye cover ho jayega", "language": "hinglish"},
    {"text": "motor insurance expire ho gaya hai renew karo jaldi", "language": "hinglish"},
    {"text": "fifty lakh ka term plan lena hai", "language": "hinglish"},
    {"text": "insurance coverage badhana hai ten lakh se twenty lakh", "language": "hinglish"},

    # FD & Savings - Hinglish
    {"text": "FD mein paisa rakhna hai safe rahega", "language": "hinglish"},
    {"text": "savings account mein balance kam hai", "language": "hinglish"},
    {"text": "RD har mahine two thousand ki hai", "language": "hinglish"},
    {"text": "FD ki interest rate kam ho gayi hai bank mein", "language": "hinglish"},
    {"text": "PPF account mein paisa dalna hai tax benefit ke liye", "language": "hinglish"},
    {"text": "minimum balance maintain karna hai savings account mein", "language": "hinglish"},
    {"text": "FD todne pe penalty lagegi premature withdrawal pe", "language": "hinglish"},
    {"text": "senior citizen FD pe extra interest milega", "language": "hinglish"},

    # Credit Card - Hinglish
    {"text": "credit card ka bill bahut zyada aaya hai is baar", "language": "hinglish"},
    {"text": "credit card limit badhana hai bank se request karo", "language": "hinglish"},
    {"text": "credit card bill ka due date kal hai pay karo", "language": "hinglish"},
    {"text": "minimum due amount bharna zaroori hai warna penalty lagegi", "language": "hinglish"},
    {"text": "credit card pe cashback mila hai two percent", "language": "hinglish"},
    {"text": "credit card se EMI pe convert karo bada transaction", "language": "hinglish"},

    # Tax - Hinglish
    {"text": "income tax return file karna hai March se pehle", "language": "hinglish"},
    {"text": "tax bachane ke liye investment karna hai ELSS mein", "language": "hinglish"},
    {"text": "section eighty C mein deduction milegi", "language": "hinglish"},
    {"text": "advance tax ka payment karna hai", "language": "hinglish"},
    {"text": "GST return file karna hai monthly basis pe", "language": "hinglish"},
    {"text": "TDS kata hai salary se check karo form sixteen", "language": "hinglish"},

    # Budget - Hinglish
    {"text": "is mahine ka budget bana lo properly", "language": "hinglish"},
    {"text": "expenses bahut badh gaye hain control karo", "language": "hinglish"},
    {"text": "saving karni hogi next month se seriously", "language": "hinglish"},
    {"text": "rent ten thousand dena hai plus maintenance charges", "language": "hinglish"},
    {"text": "electricity bill three thousand aaya hai", "language": "hinglish"},
    {"text": "grocery ka kharcha five thousand hai monthly", "language": "hinglish"},
    {"text": "salary se saving nahi ho pa rahi kuch bhi", "language": "hinglish"},
    {"text": "emergency fund banana shuru karo at least six months ka", "language": "hinglish"},
    {"text": "unnecessary expenses cut karo aur paisa save karo", "language": "hinglish"},

    # Investments - Hinglish
    {"text": "share market mein paisa lagana hai blue chip stocks mein", "language": "hinglish"},
    {"text": "gold mein invest karna safe hai generally", "language": "hinglish"},
    {"text": "property kharidne ka sahi time hai rates low hain", "language": "hinglish"},
    {"text": "dividend income acchi aa rahi hai portfolio se", "language": "hinglish"},
    {"text": "portfolio mein diversification karo risk kam hoga", "language": "hinglish"},
    {"text": "NPS mein invest karne se tax bachega extra fifty thousand", "language": "hinglish"},
    {"text": "stock sell karke profit book karo market high pe", "language": "hinglish"},
    {"text": "real estate mein invest karna risky hai abhi", "language": "hinglish"},
    {"text": "crypto mein paisa lagaya tha loss ho gaya", "language": "hinglish"},
    {"text": "sovereign gold bond le lo physical gold se better hai", "language": "hinglish"},
]

# Amount variations to inject into templates
HINDI_AMOUNTS = [
    "पाँच हज़ार", "दस हज़ार", "पंद्रह हज़ार", "बीस हज़ार", "पच्चीस हज़ार",
    "तीस हज़ार", "पचास हज़ार", "एक लाख", "दो लाख", "पाँच लाख",
    "दस लाख", "बीस लाख", "पचास लाख", "एक करोड़",
]

HINGLISH_AMOUNTS = [
    "five thousand", "ten thousand", "fifteen thousand", "twenty thousand",
    "twenty five thousand", "thirty thousand", "fifty thousand",
    "one lakh", "two lakh", "five lakh", "ten lakh", "twenty lakh",
    "fifty lakh", "one crore", "paanch hazaar", "das hazaar",
]

HINGLISH_CONNECTORS = [
    "aur", "lekin", "toh", "phir", "agar", "jab", "kyunki",
    "isliye", "matlab", "basically", "actually", "so", "but", "and",
]


# ---------------------------------------------------------------------------
# Diverse Speaker References (Librispeech 0-23)
# ---------------------------------------------------------------------------
SPEAKER_REFERENCES = [
    str(ROOT_DIR / "data/processed/openslr_librispeech_asr_clean/train.360/openslr_librispeech_asr_clean_train.360_000000.wav"),
    str(ROOT_DIR / "data/processed/openslr_librispeech_asr_clean/train.360/openslr_librispeech_asr_clean_train.360_000004.wav"),
    str(ROOT_DIR / "data/processed/openslr_librispeech_asr_clean/train.360/openslr_librispeech_asr_clean_train.360_000010.wav"),
    str(ROOT_DIR / "data/processed/openslr_librispeech_asr_clean/train.360/openslr_librispeech_asr_clean_train.360_000019.wav"),
    str(ROOT_DIR / "data/processed/openslr_librispeech_asr_clean/train.360/openslr_librispeech_asr_clean_train.360_000023.wav"),
]

# ---------------------------------------------------------------------------
# TTS Backend: Hugging Face MMS (Multilingual Massive Speech)
# Works on Python 3.12 and provides high-quality offline Hindi
# ---------------------------------------------------------------------------

_TTS_MODEL: Any = None
_TTS_TOKENIZER: Any = None

def get_tts_model():
    """Initialize and return HF MMS-TTS model (compatible with Python 3.12)."""
    global _TTS_MODEL, _TTS_TOKENIZER
    if _TTS_MODEL is None:
        from transformers import VitsModel, AutoTokenizer
        import torch
        print("\n  [TTS] Initializing Hugging Face MMS-TTS (Hindi)...")
        model_id = "facebook/mms-tts-hin"
        _TTS_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _TTS_MODEL = VitsModel.from_pretrained(model_id)
        
        # GPU Acceleration if available
        if torch.cuda.is_available():
            print("  [TTS] Using GPU (CUDA)")
            _TTS_MODEL = _TTS_MODEL.to("cuda")
        else:
            print("  [TTS] Using CPU (PyTorch)")
            
    return _TTS_MODEL, _TTS_TOKENIZER

def transliterate_hinglish_to_hindi(text: str) -> str:
    """
    Primitive but effective financial Hinglish to Devanagari mapper.
    Maps common Romanized words to Devanagari for MMS-TTS.
    """
    mapping = {
        "emi": "ईएमआई", "loan": "लोन", "sip": "एसआईपी", "tax": "टैक्स",
        "budget": "बजट", "bank": "बैंक", "insurance": "बीमा", "investment": "निवेश",
        "mutual": "म्यूचुअल", "fund": "फंड", "credit": "क्रेडिट", "card": "कार्ड",
        "bill": "बिल", "payment": "पेमेंट", "money": "पैसा", "salary": "सैलरी",
        "saving": "बचत", "expenses": "खर्चे", "profit": "मुनाफा", "loss": "नुकसान",
        "interest": "ब्याज", "rate": "दर", "return": "रिटर्न", "check": "चेक",
        "account": "खाता", "fixed": "फिक्स्ड", "deposit": "डिपॉजिट", "fd": "एफडी",
        "rd": "आरडी", "ppf": "पीपीएफ", "nps": "एनपीएस", "gold": "सोना",
        "property": "प्रॉपर्टी", "real": "रियल", "estate": "स्टेट", "stock": "स्टॉक",
        "market": "मार्केट", "share": "शेयर", "crypto": "क्रिप्टो", "bonus": "बोनस",
        "subsidy": "सब्सिडी", "policy": "पॉलिसी", "premium": "प्रीमियम", "claim": "क्लेम",
        "income": "आमदनी", "balance": "बैलेंस", "limit": "लिमिट", "due": "ड्यू",
        "date": "तारीख", "cashback": "कैशबैक", "penalty": "पेनल्टी", "return": "रिटर्न",
        "file": "फाइल", "gst": "जीएसटी", "tds": "टीडीएस", "safe": "सेफ", "next": "अगले",
        "month": "महीने", "year": "साल", "time": "समय", "control": "कंट्रोल",
        "total": "कुल", "extra": "एक्स्ट्रा", "bonus": "बोनस", "benefit": "फायदा",
        "hazaar": "हज़ार", "lakh": "लाख", "crore": "करोड़", "paanch": "पाँच", "das": "दस",
        "paisa": "पैसा", "hai": "है", "hain": "हैं", "mein": "में", "pe": "पे", "pehle": "पहले",
        "baar": "बार", "is": "इस", "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से",
        "nahi": "नहीं", "ho": "हो", "rahi": "रही", "kuch": "कुछ", "bhi": "भी", "shuru": "शुरू",
        "karo": "करो", "kar": "कर", "badh": "बढ़", "gaye": "गये", "kam": "कम", "gayi": "गयी",
        "milega": "मिलेगा", "mil": "मिल", "raha": "रहा", "thi": "थी", "tha": "था",
        "five": "पाँच", "ten": "दस", "one": "एक", "two": "दो", "three": "तीन", "four": "चार",
        "fifty": "पचास", "twenty": "बीस", "thirty": "तीस", "thousand": "हज़ार",
    }
    
    # Sort keys by length (desc) to handle composite words first
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    import re
    result = text.lower()
    for key in sorted_keys:
        # Use word boundaries (\b) to avoid partial matches
        result = re.sub(rf'\b{key}\b', mapping[key], result)
        
    return result


def generate_variations(templates: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    """Generate variations of templates to reach target count."""
    result = list(templates)
    random.seed(42)

    while len(result) < target_count:
        base = random.choice(templates)
        text = base["text"]
        lang = base["language"]

        # Create a variation by combining two templates
        other = random.choice(templates)
        if other["text"] != text:
            connector = random.choice(HINGLISH_CONNECTORS) if lang == "hinglish" else random.choice(
                ["और", "लेकिन", "तो", "फिर", "अगर", "क्योंकि", "इसलिए"]
            )
            combined = f"{text} {connector} {other['text'].lower()}"
            result.append({"text": combined, "language": lang})

    return result[:target_count]


def text_to_wav(text: str, language: str, output_path: Path, speaker_wav: str = None) -> bool:
    """Convert text to WAV using HF MMS-TTS."""
    try:
        import torch
        import scipy.io.wavfile as wavfile
        import numpy as np
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model, tokenizer = get_tts_model()
        
        # TRANS-LITERATION FOR HINGLISH (MMS-TTS requirement)
        tts_text = text
        if language == "hinglish":
            tts_text = transliterate_hinglish_to_hindi(text)

        inputs = tokenizer(tts_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        with torch.no_grad():
            output = model(**inputs).waveform
            
        # Convert to 16kHz float32 or int16
        audio_np = output.cpu().numpy().squeeze()
        # MMS-TTS output is usually 16kHz, but let's be safe
        wavfile.write(str(output_path), model.config.sampling_rate, (audio_np * 32767).astype(np.int16))
        
        return True
    except Exception as exc:
        print(f"\n  [TTS ERROR] {exc}")
        return False



def _mp3_to_wav_16k(mp3_data: io.BytesIO, output_path: Path) -> None:
    """Convert MP3 bytes to 16kHz mono WAV using imageio_ffmpeg explicitly."""
    import tempfile
    import subprocess
    import imageio_ffmpeg
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Save memory bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_data.getvalue())
        tmp_mp3_path = tmp_mp3.name

    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", tmp_mp3_path, "-ar", "16000", "-ac", "1", str(output_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"Subprocess ffmpeg failed: {e}")
        _generate_placeholder_wav(output_path)
    finally:
        try:
            import os
            os.remove(tmp_mp3_path)
        except Exception:
            pass


def _generate_placeholder_wav(output_path: Path, duration_s: float = 3.0) -> None:
    """Generate a placeholder WAV if TTS fails."""
    import math
    import wave
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    total = int(duration_s * sample_rate)
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(total):
            value = int(8000 * math.sin(2 * math.pi * 220 * i / sample_rate))
            f.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Generate Hindi/Hinglish financial TTS data.")
    parser.add_argument("--output-dir", default="data/processed/hindi_financial", help="Output directory")
    parser.add_argument("--hindi-count", type=int, default=500, help="Number of Hindi samples")
    parser.add_argument("--hinglish-count", type=int, default=700, help="Number of Hinglish samples")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip if WAV exists")
    parser.add_argument("--text-only", action="store_true", help="Generate text manifest only, no audio")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Hindi/Hinglish Financial Data Generator")
    print(f"  Target: {args.hindi_count} Hindi + {args.hinglish_count} Hinglish")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Generate variations
    hindi_data = generate_variations(HINDI_TEMPLATES, args.hindi_count)
    hinglish_data = generate_variations(HINGLISH_TEMPLATES, args.hinglish_count)
    all_data = hindi_data + hinglish_data

    print(f"\n  Generated {len(all_data)} total texts")
    if all_data:
        print(f"  First text: {all_data[0]['text']}")

    manifest_rows: list[dict[str, str]] = []
    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for idx, item in enumerate(all_data):
        text = item["text"]
        lang = item["language"]
        
        # Audio filename
        safe_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        filename = f"{'hi' if lang=='hindi' else 'hng'}_{idx:05d}_{safe_hash}.wav"
        wav_path = output_dir / filename

        if idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = (success + skipped) / max(1, elapsed) * 3600
            print(f"  [{idx}/{len(all_data)}] success={success} skipped={skipped} failed={failed} rate={rate:.0f}/h")

        if args.text_only:
            manifest_rows.append({
                "audio_path": str(wav_path),
                "text": text,
                "language": lang,
                "source": f"synthetic_{lang}_financial",
                "duration_seconds": "3.000",
            })
            success += 1
            continue

        if wav_path.exists() and args.skip_existing:
            manifest_rows.append({
                "audio_path": str(wav_path),
                "text": text,
                "language": lang,
                "source": f"synthetic_{lang}_financial",
                "duration_seconds": "3.000",
            })
            skipped += 1
            continue

        # Pick a random diverse speaker
        speaker_wav = random.choice(SPEAKER_REFERENCES)
        ok = text_to_wav(text, lang, wav_path, speaker_wav)
        if ok:
            try:
                import soundfile as sf
                info = sf.info(str(wav_path))
                duration = f"{info.duration:.3f}"
            except Exception:
                duration = "3.000"

            manifest_rows.append({
                "audio_path": str(wav_path),
                "text": text,
                "language": lang,
                "source": f"synthetic_{lang}_financial",
                "duration_seconds": duration,
            })
            success += 1
        else:
            failed += 1
            # If we recently had a long string of failures, maybe the API is totally blocked
            if failed > 50:
                 print("\n  Too many consecutive failures. Consider reducing counts or waiting.")

    # Merge manifest — read existing entries first, then append new ones (no overwrite)
    manifest_path = output_dir / "hindi_financial_manifest.csv"
    existing_rows: dict[str, dict] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows[row["audio_path"]] = row

    # Merge: new rows override existing by audio_path
    for row in manifest_rows:
        existing_rows[row["audio_path"]] = row

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text", "language", "source", "duration_seconds"])
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"  Manifest has {len(existing_rows)} total rows (merged)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed / 60:.1f} minutes")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
