"""
=============================================================================
EXPERT AI MACHINE TRANSLATION SYSTEM — v3.0 (Groq AI Edition)
=============================================================================
Author  : [Student Name] | Reg No: [Registration Number] | Dept: [Department]
Project : Neural-inspired Expert Machine Translation Web Application
=============================================================================

AI MODELS & TECHNIQUES APPLIED:
────────────────────────────────
1. CORPUS-BASED MACHINE TRANSLATION (CBMT) — EBMT Exact Lookup
2. TF-IDF + COSINE SIMILARITY — Statistical Vector Space Model
3. N-GRAM LANGUAGE MODEL — Bigram Fluency Scoring
4. TOKEN OVERLAP / JACCARD SIMILARITY — Fuzzy Fallback
5. LANGUAGE DETECTION — Unicode Block + Stopword Heuristics
6. GROQ AI (LLaMA 3.3 70B) — PRIMARY Neural Fallback (FREE tier)
7. MyMemory FREE API — SECONDARY Backup Fallback

Fallback Chain:
    Corpus Exact → TF-IDF Cosine → Jaccard Fuzzy
    → Groq AI (expert quality) → MyMemory (free backup)
=============================================================================
"""

import os, re, ast, math, json, time, random, logging
import urllib.request, urllib.parse, urllib.error
from collections import defaultdict, Counter

import pandas as pd
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_ENSURE_ASCII"] = False

# ─── Student Info ─────────────────────────────────────────────────────────────
STUDENT_INFO = {
    "full_name"  : "EZAEKOR CHUKWUEBUKA EMMANUEL",
    "reg_number" : "20231381262",
    "department" : "CYBERSECURITY",
    "institution": "FEDERAL UNIVERSITY OF TECHNOLOGY OWERRI",
    "session"    : "2025/2026",
    "project"    : "Expert AI Machine Translation System v3.0",
}

# ─── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY       = "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"  # paste key here or set env var
MYMEMORY_EMAIL     = "ebukaemma787@gmail.com"
LIBRETRANSLATE_URL = os.environ.get("LIBRETRANSLATE_URL", "")

# ─── Language Metadata ────────────────────────────────────────────────────────
LANGUAGE_META = {
    "en": {"name": "English",    "flag": "GB", "rtl": False, "voice_lang": "en-US", "mm": "en"},
    "fr": {"name": "French",     "flag": "FR", "rtl": False, "voice_lang": "fr-FR", "mm": "fr"},
    "de": {"name": "German",     "flag": "DE", "rtl": False, "voice_lang": "de-DE", "mm": "de"},
    "es": {"name": "Spanish",    "flag": "ES", "rtl": False, "voice_lang": "es-ES", "mm": "es"},
    "it": {"name": "Italian",    "flag": "IT", "rtl": False, "voice_lang": "it-IT", "mm": "it"},
    "nl": {"name": "Dutch",      "flag": "NL", "rtl": False, "voice_lang": "nl-NL", "mm": "nl"},
    "ru": {"name": "Russian",    "flag": "RU", "rtl": False, "voice_lang": "ru-RU", "mm": "ru"},
    "hu": {"name": "Hungarian",  "flag": "HU", "rtl": False, "voice_lang": "hu-HU", "mm": "hu"},
    "pt": {"name": "Portuguese", "flag": "PT", "rtl": False, "voice_lang": "pt-PT", "mm": "pt"},
    "pl": {"name": "Polish",     "flag": "PL", "rtl": False, "voice_lang": "pl-PL", "mm": "pl"},
    "no": {"name": "Norwegian",  "flag": "NO", "rtl": False, "voice_lang": "nb-NO", "mm": "no"},
    "sv": {"name": "Swedish",    "flag": "SE", "rtl": False, "voice_lang": "sv-SE", "mm": "sv"},
    "fi": {"name": "Finnish",    "flag": "FI", "rtl": False, "voice_lang": "fi-FI", "mm": "fi"},
    "el": {"name": "Greek",      "flag": "GR", "rtl": False, "voice_lang": "el-GR", "mm": "el"},
    "ca": {"name": "Catalan",    "flag": "ES", "rtl": False, "voice_lang": "ca-ES", "mm": "ca"},
    "eo": {"name": "Esperanto",  "flag": "EO", "rtl": False, "voice_lang": "eo",    "mm": "eo"},
    "zh": {"name": "Chinese",    "flag": "CN", "rtl": False, "voice_lang": "zh-CN", "mm": "zh"},
    "ja": {"name": "Japanese",   "flag": "JP", "rtl": False, "voice_lang": "ja-JP", "mm": "ja"},
    "ar": {"name": "Arabic",     "flag": "SA", "rtl": True,  "voice_lang": "ar-SA", "mm": "ar"},
    "ko": {"name": "Korean",     "flag": "KR", "rtl": False, "voice_lang": "ko-KR", "mm": "ko"},
    "tr": {"name": "Turkish",    "flag": "TR", "rtl": False, "voice_lang": "tr-TR", "mm": "tr"},
    "hi": {"name": "Hindi",      "flag": "IN", "rtl": False, "voice_lang": "hi-IN", "mm": "hi"},
    "yo": {"name": "Yoruba",     "flag": "NG", "rtl": False, "voice_lang": "yo",    "mm": "yo"},
    "ig": {"name": "Igbo",       "flag": "NG", "rtl": False, "voice_lang": "ig",    "mm": "ig"},
    "ha": {"name": "Hausa",      "flag": "NG", "rtl": False, "voice_lang": "ha",    "mm": "ha"},
    "sw": {"name": "Swahili",    "flag": "KE", "rtl": False, "voice_lang": "sw-KE", "mm": "sw"},
    "ro": {"name": "Romanian",   "flag": "RO", "rtl": False, "voice_lang": "ro-RO", "mm": "ro"},
    "cs": {"name": "Czech",      "flag": "CZ", "rtl": False, "voice_lang": "cs-CZ", "mm": "cs"},
    "uk": {"name": "Ukrainian",  "flag": "UA", "rtl": False, "voice_lang": "uk-UA", "mm": "uk"},
    "vi": {"name": "Vietnamese", "flag": "VN", "rtl": False, "voice_lang": "vi-VN", "mm": "vi"},
    "id": {"name": "Indonesian", "flag": "ID", "rtl": False, "voice_lang": "id-ID", "mm": "id"},
    "th": {"name": "Thai",       "flag": "TH", "rtl": False, "voice_lang": "th-TH", "mm": "th"},
}

CORPUS       = {}
TFIDF_INDEX  = {}
NGRAM_MODELS = defaultdict(Counter)
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")

# ─── Built-in Phrasebook (no dataset needed) ─────────────────────────────────
PHRASEBOOK = {
    "greetings": [
        "Hello, how are you?",
        "Good morning! Have a great day.",
        "Good evening! How was your day?",
        "Nice to meet you.",
        "Goodbye, take care!",
        "How is your family?",
        "Welcome! Please come in.",
        "See you tomorrow.",
    ],
    "travel": [
        "Where is the nearest hotel?",
        "How much does this cost?",
        "Can you help me please?",
        "I need a taxi to the airport.",
        "Where is the bus station?",
        "I have lost my way.",
        "What time does the train leave?",
        "Do you speak English?",
    ],
    "food & dining": [
        "I am very hungry.",
        "What do you recommend here?",
        "The food is absolutely delicious!",
        "Can I see the menu please?",
        "I am a vegetarian.",
        "A glass of water please.",
        "This is too spicy for me.",
        "Can I have the bill please?",
    ],
    "emergency": [
        "Please call the police!",
        "I need to see a doctor immediately.",
        "Help me, please!",
        "There is a fire in the building.",
        "I have lost my passport.",
        "Please call an ambulance.",
        "This is an emergency!",
        "Where is the nearest hospital?",
    ],
    "academic": [
        "I am a university student.",
        "I do not understand this topic.",
        "Can you please explain that again?",
        "What is the meaning of this word?",
        "I need to prepare for my exam.",
        "When is the assignment due?",
        "This lecture is very interesting.",
        "I am studying computer science.",
    ],
    "nigerian expressions": [
        "How body? (How are you?)",
        "I dey fine. (I am fine.)",
        "E don do. (It is finished.)",
        "No wahala. (No problem.)",
        "Na so e be. (That is how it is.)",
        "Oya let us go. (Let us leave now.)",
        "You too much! (You are amazing!)",
        "Na God dey do am. (God is in control.)",
    ],
}

# ─── Language Fun Facts ───────────────────────────────────────────────────────
LANGUAGE_FACTS = [
    {"fact": "There are over 7,000 languages spoken in the world today, but more than half are endangered.", "emoji": "🌍"},
    {"fact": "Mandarin Chinese is the most spoken language by native speakers, with over 900 million speakers.", "emoji": "🇨🇳"},
    {"fact": "Nigeria is the most linguistically diverse country in Africa, with over 500 distinct languages.", "emoji": "🇳🇬"},
    {"fact": "Igbo, spoken in southeastern Nigeria, has over 120 dialects across different communities.", "emoji": "📜"},
    {"fact": "Hausa is one of the most widely spoken languages in Africa, with over 70 million native speakers.", "emoji": "🌍"},
    {"fact": "Yoruba is a tonal language — the same word can mean completely different things depending on your pitch.", "emoji": "🎵"},
    {"fact": "Japanese has three writing systems: Hiragana, Katakana, and Kanji — all used together.", "emoji": "🇯🇵"},
    {"fact": "Arabic is written right to left and is the official language of 26 countries across the world.", "emoji": "🇸🇦"},
    {"fact": "The longest word in German is 'Donaudampfschiffahrtsgesellschaftskapitän' — meaning 'Danube steamship company captain'.", "emoji": "🇩🇪"},
    {"fact": "Esperanto was invented in 1887 by L.L. Zamenhof as a universal language for world peace.", "emoji": "🌐"},
    {"fact": "Swahili is spoken by over 200 million people across East Africa, making it the most spoken Bantu language.", "emoji": "🌍"},
    {"fact": "Over half of the world's population is bilingual or multilingual — monolingualism is actually the minority!", "emoji": "🧠"},
    {"fact": "The first machine translation system was demonstrated at Georgetown University in 1954 — it translated 60 Russian sentences into English.", "emoji": "🤖"},
    {"fact": "TF-IDF was invented in the 1970s and remains one of the most important algorithms in information retrieval and NLP.", "emoji": "📊"},
    {"fact": "Greek is one of the oldest living languages, with written records going back over 3,400 years.", "emoji": "🇬🇷"},
    {"fact": "The word 'robot' comes from the Czech word 'robota', meaning 'forced labor' — first used in a 1920 play.", "emoji": "🤖"},
    {"fact": "South Africa has 11 official languages — the most of any country in the world.", "emoji": "🏳️"},
    {"fact": "Finnish and Hungarian are unique in Europe — they are unrelated to all neighboring languages, belonging to the Uralic family.", "emoji": "🧩"},
    {"fact": "The translation industry generates over $56 billion in revenue globally every year.", "emoji": "💼"},
    {"fact": "LLaMA (Large Language Model Meta AI), used in this system via Groq, represents the cutting edge of open-source neural MT.", "emoji": "🦙"},
]


# =============================================================================
# TEXT PRE-PROCESSING
# =============================================================================
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'\-]", " ", text)
    return re.sub(r"\s+", " ", text)

def tokenize(text):
    return [w for w in normalize_text(text).split() if w]

def build_ngrams(tokens, n=2):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# =============================================================================
# CORPUS LOADING
# =============================================================================
def parse_translation_cell(cell):
    try:
        r = ast.literal_eval(cell)
        return r if isinstance(r, dict) else {}
    except (ValueError, SyntaxError):
        return {}

def load_corpus_file(filepath, src_lang, tgt_lang, max_rows=15000):
    pairs = []
    try:
        df = pd.read_csv(filepath, nrows=max_rows)
        for _, row in df.iterrows():
            t = parse_translation_cell(str(row.get("translation", "")))
            s, g = t.get(src_lang, "").strip(), t.get(tgt_lang, "").strip()
            if len(s) >= 3 and len(g) >= 3:
                pairs.append((s, g))
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
    return pairs

def load_all_corpora():
    logger.info("Loading parallel corpora...")
    if not os.path.exists(DATA_DIR):
        logger.warning(f"No data dir — Groq AI handles all {len(LANGUAGE_META)} languages.")
        return
    for filename in [f for f in os.listdir(DATA_DIR) if f.endswith("_train.csv")]:
        parts = filename.replace("_train.csv", "").split("-")
        if len(parts) != 2: continue
        la, lb = parts
        if la not in LANGUAGE_META or lb not in LANGUAGE_META: continue
        fp = os.path.join(DATA_DIR, filename)
        ab = load_corpus_file(fp, la, lb)
        if ab:
            CORPUS[(la, lb)] = ab
            CORPUS[(lb, la)] = [(t, s) for s, t in ab]
            for pairs in [ab, CORPUS[(lb, la)]]:
                for _, tgt in pairs:
                    for bg in build_ngrams(tokenize(tgt)):
                        NGRAM_MODELS[lb if pairs is ab else la][bg] += 1
            logger.info(f"  {la}<->{lb}: {len(ab):,} pairs")
    logger.info(f"Corpus: {len(CORPUS)} pairs loaded")


# =============================================================================
# TF-IDF
# =============================================================================
def compute_tf(tokens):
    total = len(tokens)
    if not total: return {}
    freq = Counter(tokens)
    return {t: c / total for t, c in freq.items()}

def compute_idf(corpus_tokens):
    N, df = len(corpus_tokens), defaultdict(int)
    for doc in corpus_tokens:
        for t in set(doc): df[t] += 1
    return {t: math.log(N / c) + 1.0 for t, c in df.items()}

def tfidf_vector(tokens, idf):
    tf = compute_tf(tokens)
    return {t: v * idf.get(t, 1.0) for t, v in tf.items()}

def cosine_similarity(va, vb):
    dot = sum(va.get(t, 0.0) * v for t, v in vb.items())
    na  = math.sqrt(sum(v**2 for v in va.values()))
    nb  = math.sqrt(sum(v**2 for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0

def build_tfidf_index(lang_pair):
    if lang_pair in TFIDF_INDEX or not CORPUS.get(lang_pair): return
    pairs      = CORPUS[lang_pair]
    all_tokens = [tokenize(s) for s, _ in pairs]
    idf        = compute_idf(all_tokens)
    TFIDF_INDEX[lang_pair] = {"idf": idf, "vectors": [tfidf_vector(t, idf) for t in all_tokens]}


# =============================================================================
# GROQ AI — PRIMARY NEURAL FALLBACK
# =============================================================================
def translate_groq(query: str, src_lang: str, tgt_lang: str):
    """
    Groq AI — LLaMA 3.3 70B. Expert-quality neural translation.
    Free tier at console.groq.com. Ultra-fast inference.
    Best for African languages (Yoruba, Igbo, Hausa) and all rare pairs.
    Returns None on network failure (caller tries next fallback).
    """
    t0       = time.perf_counter()
    src_name = LANGUAGE_META.get(src_lang, {}).get("name", src_lang)
    tgt_name = LANGUAGE_META.get(tgt_lang, {}).get("name", tgt_lang)

    payload = {
        "model"      : "llama-3.3-70b-versatile",
        "max_tokens" : 1024,
        "temperature": 0.1,
        "messages"   : [
            {"role": "system", "content": (
                "You are an expert linguist and professional translator. "
                "Rules: (1) Return ONLY the translated text — no notes, no explanations. "
                "(2) Preserve tone and meaning exactly. "
                "(3) For Yoruba, Igbo, Hausa: use correct diacritics and tones. "
                "(4) Do not add quotation marks around the translation."
            )},
            {"role": "user", "content": f"Translate from {src_name} to {tgt_name}:\n\n{query}"},
        ],
    }
    try:
        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        translated = data["choices"][0]["message"]["content"].strip().strip('"').strip("'")
        elapsed    = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Groq: '{query[:30]}' -> '{translated[:30]}'")
        return {
            "translation": translated,
            "confidence" : 95,
            "method"     : f"Groq AI — LLaMA 3.3 70B [{src_name}→{tgt_name}]",
            "src_sample" : query,
            "time_ms"    : elapsed,
        }
    except urllib.error.HTTPError as e:
        code = e.code
        logger.error(f"Groq HTTP {code}")
        if code == 401:
            return {"translation": "⚠ Invalid Groq API key. Please check your GROQ_API_KEY.",
                    "confidence": 0, "method": "Groq Auth Error", "src_sample": query, "time_ms": 0}
        if code == 429:
            return None  # rate limit — try MyMemory
        return None
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return None


# =============================================================================
# MyMemory — SECONDARY BACKUP
# =============================================================================
def translate_mymemory(query, src_lang, tgt_lang):
    t0    = time.perf_counter()
    s_mm  = LANGUAGE_META.get(src_lang, {}).get("mm", src_lang)
    t_mm  = LANGUAGE_META.get(tgt_lang, {}).get("mm", tgt_lang)
    params = {"q": query, "langpair": f"{s_mm}|{t_mm}"}
    if MYMEMORY_EMAIL: params["de"] = MYMEMORY_EMAIL
    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ExpertMT/3.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        status = data.get("responseStatus", 0)
        if status in (200, 206):
            translated = data["responseData"]["translatedText"].strip()
            match_val  = float(data["responseData"].get("match", 0.8))
            if translated.lower() == query.lower():
                return {"translation": "No translation found for this pair.",
                        "confidence": 0, "method": "MyMemory (no result)", "src_sample": "", "time_ms": 0}
            return {"translation": translated,
                    "confidence": min(85, int(0.5 * match_val * 100 + 45)),
                    "method"    : f"MyMemory API (backup) [match={match_val:.2f}]",
                    "src_sample": query, "time_ms": round((time.perf_counter()-t0)*1000, 2)}
        elif status == 403:
            return {"translation": "Daily free quota exceeded. Try again tomorrow.",
                    "confidence": 0, "method": "MyMemory Quota Exceeded", "src_sample": query, "time_ms": 0}
        return {"translation": f"API error status {status}.",
                "confidence": 0, "method": f"MyMemory Error", "src_sample": "", "time_ms": 0}
    except Exception as e:
        logger.error(f"MyMemory error: {e}")
        return {"translation": "Network error. Check your connection.",
                "confidence": 0, "method": "Network Error", "src_sample": "", "time_ms": 0}

def translate_libretranslate(query, src_lang, tgt_lang):
    t0      = time.perf_counter()
    payload = json.dumps({"q": query, "source": src_lang, "target": tgt_lang, "format": "text"}).encode()
    try:
        req = urllib.request.Request(f"{LIBRETRANSLATE_URL.rstrip('/')}/translate",
                                     data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        t = data.get("translatedText", "").strip()
        if not t: return None
        return {"translation": t, "confidence": 85, "method": "LibreTranslate (Local)",
                "src_sample": query, "time_ms": round((time.perf_counter()-t0)*1000, 2)}
    except Exception as e:
        logger.warning(f"LibreTranslate unavailable: {e}")
        return None


# =============================================================================
# MAIN TRANSLATION PIPELINE
# =============================================================================
def jaccard_similarity(a, b):
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0

def ngram_confidence_score(translation, tgt_lang):
    tokens = tokenize(translation)
    if len(tokens) < 2: return 0.5
    bigrams = build_ngrams(tokens)
    if not bigrams: return 0.5
    model = NGRAM_MODELS.get(tgt_lang, Counter())
    if not model: return 0.5
    return sum(1 for bg in bigrams if model[bg] > 0) / len(bigrams)

def translate(query, src_lang, tgt_lang):
    t0        = time.perf_counter()
    query     = query.strip()
    lang_pair = (src_lang, tgt_lang)
    pairs     = CORPUS.get(lang_pair)

    if pairs:
        qnorm = normalize_text(query)
        qtoks = tokenize(query)

        # Step 1: Exact Match
        if not hasattr(translate, "_cache"): translate._cache = {}
        ck = f"exact_{lang_pair}"
        if ck not in translate._cache:
            translate._cache[ck] = {normalize_text(s): t for s, t in pairs}
        if qnorm in translate._cache[ck]:
            tgt = translate._cache[ck][qnorm]
            fl  = ngram_confidence_score(tgt, tgt_lang)
            return {"translation": tgt, "confidence": min(100, int((0.95+0.05*fl)*100)),
                    "method": "Exact Match (EBMT — Corpus)", "src_sample": query,
                    "time_ms": round((time.perf_counter()-t0)*1000, 2)}

        # Step 2: TF-IDF Cosine
        build_tfidf_index(lang_pair)
        best_cos, best_idx = 0.0, -1
        if lang_pair in TFIDF_INDEX:
            idf, vectors = TFIDF_INDEX[lang_pair]["idf"], TFIDF_INDEX[lang_pair]["vectors"]
            qvec = tfidf_vector(qtoks, idf)
            for i, cv in enumerate(vectors):
                s = cosine_similarity(qvec, cv)
                if s > best_cos: best_cos, best_idx = s, i
        if best_cos >= 0.35 and best_idx >= 0:
            sm, tm = pairs[best_idx]
            fl = ngram_confidence_score(tm, tgt_lang)
            return {"translation": tm, "confidence": min(98, int((0.7*best_cos+0.3*fl)*100)),
                    "method": f"TF-IDF + Cosine (SMT) [cos={best_cos:.2f}]",
                    "src_sample": sm, "time_ms": round((time.perf_counter()-t0)*1000, 2)}

        # Step 3: Jaccard Fuzzy
        best_jac, best_jidx = 0.0, -1
        for i, (src, _) in enumerate(pairs):
            j = jaccard_similarity(qtoks, tokenize(src))
            if j > best_jac: best_jac, best_jidx = j, i
        if best_jac >= 0.20 and best_jidx >= 0:
            sm, tm = pairs[best_jidx]
            fl = ngram_confidence_score(tm, tgt_lang)
            return {"translation": tm, "confidence": min(80, int((0.6*best_jac+0.4*fl)*100)),
                    "method": f"Jaccard Fuzzy Match (EBMT) [jac={best_jac:.2f}]",
                    "src_sample": sm, "time_ms": round((time.perf_counter()-t0)*1000, 2)}

    # Step 4: LibreTranslate (optional)
    if LIBRETRANSLATE_URL:
        r = translate_libretranslate(query, src_lang, tgt_lang)
        if r: return r

    # Step 5: Groq AI — PRIMARY neural (expert quality)
    if GROQ_API_KEY:
        r = translate_groq(query, src_lang, tgt_lang)
        if r and r.get("confidence", 0) > 0: return r

    # Step 6: MyMemory — free backup
    return translate_mymemory(query, src_lang, tgt_lang)


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================
CYRILLIC=(0x0400,0x04FF); GREEK=(0x0370,0x03FF); ARABIC=(0x0600,0x06FF)
CJK=(0x4E00,0x9FFF);      HANGUL=(0xAC00,0xD7AF); DEVANAG=(0x0900,0x097F)
THAI=(0x0E00,0x0E7F)

LANG_MARKERS = {
    "en": {"the","is","are","and","of","to","a","in","it","that","was","for"},
    "fr": {"le","la","les","de","et","un","une","est","dans","je","pas","sur"},
    "de": {"die","der","das","und","ist","ein","eine","ich","nicht","mit","zu"},
    "es": {"el","la","los","de","y","en","que","un","una","es","se","por"},
    "it": {"il","la","le","di","e","un","una","che","non","si","per"},
    "nl": {"de","het","een","van","en","in","is","dat","op","te","zijn"},
    "hu": {"a","az","es","van","nem","hogy","egy","is","ez","de"},
    "pt": {"o","a","os","as","de","e","um","uma","em","que","nao","para"},
    "pl": {"i","w","z","nie","na","to","sie","jest","ze","do"},
    "no": {"og","i","er","det","en","et","som","av","til","pa"},
    "sv": {"och","ar","i","det","en","ett","att","av","for","med"},
    "fi": {"ja","on","ei","se","ole","olla","etta","han","ne","kun"},
    "eo": {"kaj","la","de","en","al","ne","estas","mi","li","sed"},
    "tr": {"ve","bir","bu","da","de","ile","icin","var","ne","ki"},
    "sw": {"na","ya","wa","la","ni","kwa","si","au","katika","hii"},
    "yo": {"ti","ni","bi","ati","fun","naa","le","pe","se"},
    "ha": {"da","na","ba","shi","ne","kai","sai","amma","ko"},
    "ig": {"na","ya","n","ka","nke","o","ha","di","nwere"},
    "ro": {"si","in","de","ca","este","nu","cu","la","din","pentru"},
    "cs": {"a","v","na","je","to","se","ze","s","do","z"},
    "vi": {"va","cua","la","co","trong","duoc","toi","khong","voi"},
    "id": {"dan","yang","di","ini","itu","dengan","untuk","tidak","ada"},
}

def detect_language(text):
    tokens = set(tokenize(text.lower()))
    def cr(lo, hi): return sum(1 for ch in text if lo <= ord(ch) <= hi)
    if cr(*CYRILLIC)>3: return "ru"
    if cr(*GREEK)>3:    return "el"
    if cr(*ARABIC)>3:   return "ar"
    if cr(*CJK)>3:      return "zh"
    if cr(*HANGUL)>3:   return "ko"
    if cr(*DEVANAG)>3:  return "hi"
    if cr(*THAI)>3:     return "th"
    scores = {lang: len(tokens & m) for lang, m in LANG_MARKERS.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"


# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route("/")
def index():
    available = []
    for (src, tgt) in CORPUS:
        sm, tm = LANGUAGE_META.get(src,{}), LANGUAGE_META.get(tgt,{})
        available.append({"code": f"{src}-{tgt}", "src": src, "tgt": tgt,
                          "src_name": sm.get("name",src.upper()), "tgt_name": tm.get("name",tgt.upper()),
                          "src_flag": sm.get("flag",""), "tgt_flag": tm.get("flag","")})
    available.sort(key=lambda x: x["src_name"])
    return render_template("index.html",
                           student=STUDENT_INFO, languages=LANGUAGE_META, pairs=available,
                           groq_active=bool(GROQ_API_KEY), libretranslate_active=bool(LIBRETRANSLATE_URL))

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data     = request.get_json(force=True, silent=True) or {}
    text     = str(data.get("text","")).strip()
    src_lang = str(data.get("src","en")).strip().lower()
    tgt_lang = str(data.get("tgt","fr")).strip().lower()
    if not text: return jsonify({"error": "Input text is required"}), 400
    if src_lang == tgt_lang:
        return jsonify({"translation": text, "confidence": 100, "method": "Same language",
                        "src_sample": text, "time_ms": 0})
    return jsonify(translate(text, src_lang, tgt_lang))

@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text","")).strip()
    if not text: return jsonify({"error": "Text is required"}), 400
    lang = detect_language(text)
    meta = LANGUAGE_META.get(lang, {"name": lang.upper(), "flag": ""})
    return jsonify({"lang": lang, "name": meta["name"], "flag": meta.get("flag","")})

@app.route("/api/pairs")
def api_pairs():
    pairs = []
    for (src,tgt), sl in CORPUS.items():
        sm, tm = LANGUAGE_META.get(src,{}), LANGUAGE_META.get(tgt,{})
        pairs.append({"src": src, "tgt": tgt,
                      "src_name": sm.get("name",src), "tgt_name": tm.get("name",tgt),
                      "src_flag": sm.get("flag",""), "tgt_flag": tm.get("flag",""), "count": len(sl)})
    pairs.sort(key=lambda x: (x["src_name"], x["tgt_name"]))
    return jsonify(pairs)

@app.route("/api/stats")
def api_stats():
    n = len(LANGUAGE_META)
    corpus_langs = set(l for pair in CORPUS for l in pair)
    return jsonify({
        "total_pairs"          : len(CORPUS) or (n*(n-1)),
        "total_languages"      : n,
        "total_sentences"      : sum(len(v) for v in CORPUS.values()),
        "corpus_pairs"         : len(CORPUS),
        "languages"            : sorted(LANGUAGE_META.keys()),
        "groq_active"          : bool(GROQ_API_KEY),
        "free_api_active"      : True,
        "libretranslate_active": bool(LIBRETRANSLATE_URL),
        "api_powered"          : not bool(CORPUS),
    })

@app.route("/api/phrasebook")
def api_phrasebook():
    return jsonify({"categories": list(PHRASEBOOK.keys()), "phrases": PHRASEBOOK})

@app.route("/api/funfact")
def api_funfact():
    return jsonify(random.choice(LANGUAGE_FACTS))

@app.route("/api/sample/<src>/<tgt>")
def api_sample(src, tgt):
    pairs = CORPUS.get((src, tgt), [])
    if not pairs: return jsonify({"samples": []}), 404
    samples = random.sample(pairs, min(5, len(pairs)))
    return jsonify({"samples": [{"src": s, "tgt": t} for s, t in samples]})

@app.route("/api/student")
def api_student():
    return jsonify(STUDENT_INFO)

@app.route("/api/student/update", methods=["POST"])
def api_student_update():
    data = request.get_json(force=True, silent=True) or {}
    for f in ["full_name","reg_number","department","institution","session"]:
        if f in data and str(data[f]).strip():
            STUDENT_INFO[f] = str(data[f]).strip()
    return jsonify({"status": "ok", "student": STUDENT_INFO})

@app.route("/api/language_meta")
def api_language_meta():
    return jsonify(LANGUAGE_META)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("""
+--------------------------------------------------------------+
|   EXPERT AI MACHINE TRANSLATION SYSTEM  v3.0                 |
|   EBMT · TF-IDF · N-gram · Groq AI LLaMA 3.3 · MyMemory     |
+--------------------------------------------------------------+""")
    load_all_corpora()
    print("\n── AI Status ───────────────────────────────────────────────")
    if GROQ_API_KEY:
        print("✅  Groq AI      : Active  (LLaMA 3.3 70B — expert quality)")
    else:
        print("⚠   Groq AI      : No key  (add GROQ_API_KEY for best results)")
        print("    Get free key : https://console.groq.com → API Keys")
    print("✅  MyMemory     : Active  (free backup, 5,000 chars/day)")
    if LIBRETRANSLATE_URL:
        print(f"✅  LibreTranslate: {LIBRETRANSLATE_URL}")
    print(f"\n── Corpus ──────────────────────────────────────────────────")
    if CORPUS:
        print(f"✅  {len(CORPUS)} pairs, {sum(len(v) for v in CORPUS.values()):,} sentences")
    else:
        print(f"ℹ   No dataset — Groq AI covers all {len(LANGUAGE_META)} languages")
    print(f"\n🌐  Server: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)