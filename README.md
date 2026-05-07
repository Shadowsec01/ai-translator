# 🌍 Expert AI Machine Translation System v3.0

A neural-inspired multilingual translation web application built with Flask and Python. Developed as an academic project at the Federal University of Technology Owerri (FUTO), this system combines classical NLP techniques with modern AI to deliver high-quality translations across 32 languages — including Nigerian indigenous languages.

---

## 👨‍💻 Author

| Field | Detail |
|---|---|
| **Name** | Ezaekor Chukwuebuka Emmanuel |
| **Reg Number** | 20231381262 |
| **Department** | Cybersecurity |
| **Institution** | Federal University of Technology Owerri (FUTO) |
| **Session** | 2025/2026 |

---

## 🚀 Features

- **5-stage intelligent translation pipeline** with automatic fallback
- **32 supported languages** including Yoruba, Igbo, and Hausa
- **Automatic language detection** using Unicode block analysis and stopword heuristics
- **Built-in Phrasebook** with categorized common phrases (greetings, travel, emergency, academic, Nigerian expressions)
- **Language Fun Facts** panel — random linguistic facts served via API
- **Corpus-based translation** with TF-IDF indexing for fast local lookups
- **Groq AI (LLaMA 3.3 70B)** as primary neural fallback for expert-quality translations
- **MyMemory API** as secondary backup fallback
- **Student profile configuration** via API endpoint
- Deployed on **Render.com**

---

## 🧠 AI Models & Techniques

The translation pipeline runs through 5 stages in order, falling back to the next if the current stage fails or returns a low-confidence result:

```
1. Corpus Exact Match (EBMT)
        ↓
2. TF-IDF + Cosine Similarity (Statistical Vector Space Model)
        ↓
3. Jaccard Fuzzy Match (Token Overlap)
        ↓
4. Groq AI — LLaMA 3.3 70B (Primary Neural Fallback)
        ↓
5. MyMemory API (Secondary Backup Fallback)
```

| Technique | Description |
|---|---|
| **EBMT Exact Lookup** | Direct corpus match using normalized text |
| **TF-IDF + Cosine Similarity** | Vector space model for fuzzy corpus retrieval |
| **N-gram Language Model** | Bigram fluency scoring for confidence estimation |
| **Jaccard Similarity** | Token overlap fuzzy fallback |
| **Language Detection** | Unicode block + stopword heuristics |
| **Groq AI (LLaMA 3.3 70B)** | Expert-quality neural translation via free Groq API |
| **MyMemory API** | Free backup translation API (5,000 chars/day) |

---

## 🌐 Supported Languages

| Code | Language | Code | Language |
|---|---|---|---|
| `en` | English | `zh` | Chinese |
| `fr` | French | `ja` | Japanese |
| `de` | German | `ar` | Arabic |
| `es` | Spanish | `ko` | Korean |
| `it` | Italian | `tr` | Turkish |
| `nl` | Dutch | `hi` | Hindi |
| `ru` | Russian | `yo` | Yoruba 🇳🇬 |
| `pt` | Portuguese | `ig` | Igbo 🇳🇬 |
| `pl` | Polish | `ha` | Hausa 🇳🇬 |
| `sv` | Swedish | `sw` | Swahili |
| `no` | Norwegian | `ro` | Romanian |
| `fi` | Finnish | `cs` | Czech |
| `el` | Greek | `uk` | Ukrainian |
| `hu` | Hungarian | `vi` | Vietnamese |
| `ca` | Catalan | `id` | Indonesian |
| `eo` | Esperanto | `th` | Thai |

---

## 🛠️ Tech Stack

- **Backend:** Python 3, Flask
- **NLP:** Custom TF-IDF, Cosine Similarity, N-gram, Jaccard — all implemented from scratch
- **AI:** Groq API (LLaMA 3.3 70B), MyMemory API
- **Data:** pandas for corpus loading
- **Frontend:** HTML/CSS/JS (single template)
- **Deployment:** Render.com

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- A free [Groq API key](https://console.groq.com) (for AI translation)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-translation-system.git
cd ai-translation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

Either set it as an environment variable:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

Or paste it directly into `app.py`:
```python
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. (Optional) Add a corpus dataset

Place parallel corpus CSV files in a `data/` folder with the naming format:
```
data/en-fr_train.csv
data/en-de_train.csv
```

Each CSV must have a `translation` column containing a dictionary with language codes as keys:
```
{"en": "Hello", "fr": "Bonjour"}
```

If no corpus is provided, the system runs entirely on Groq AI and MyMemory API.

### 5. Run the application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 🌍 Deployment (Render.com)

This project is configured for one-click deployment on Render.com via `render.yaml`.

1. Push the project to a GitHub repository
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repo
4. Render will automatically detect `render.yaml` and deploy

Make sure to set `GROQ_API_KEY` as an environment variable in your Render dashboard under **Environment**.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web interface |
| `POST` | `/api/translate` | Translate text |
| `POST` | `/api/detect` | Detect language of input text |
| `GET` | `/api/pairs` | List all available corpus language pairs |
| `GET` | `/api/stats` | System statistics |
| `GET` | `/api/phrasebook` | Get all phrasebook categories and phrases |
| `GET` | `/api/funfact` | Get a random language fun fact |
| `GET` | `/api/language_meta` | Get metadata for all supported languages |
| `GET` | `/api/student` | Get student profile info |
| `POST` | `/api/student/update` | Update student profile info |

### Example: Translate text
```bash
curl -X POST http://localhost:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "src": "en", "tgt": "yo"}'
```

**Response:**
```json
{
  "translation": "Bawo ni o ṣe wa?",
  "confidence": 95,
  "method": "Groq AI — LLaMA 3.3 70B [English→Yoruba]",
  "src_sample": "Hello, how are you?",
  "time_ms": 843.21
}
```

---

## 📁 Project Structure

```
ai-translation-system/
│
├── app.py              # Main Flask application and translation pipeline
├── requirements.txt    # Python dependencies
├── render.yaml         # Render.com deployment configuration
├── templates/
│   └── index.html      # Frontend web interface
└── data/               # (Optional) Parallel corpus CSV files
```

---

## 📌 Notes

- The system works **without any corpus** — Groq AI handles all 32 languages automatically
- Nigerian indigenous languages (Yoruba, Igbo, Hausa) are best handled by Groq AI with correct diacritics
- MyMemory free tier allows 5,000 characters per day — suitable for low-volume use
- The `GROQ_API_KEY` in the source file should be replaced with an environment variable before publishing publicly

---

## 📄 License

This project was developed for academic purposes at FUTO. All rights reserved by the author.
