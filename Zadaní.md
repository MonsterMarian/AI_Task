- **Řešení:** Využij Pydantic modely ve FastAPI pro validaci requestu i response. V promptu explicitně zakaž jakýkoliv jiný text kromě čisté JSON struktury. Pokud zvolíš bonusové streamování, endpoint musí správně složit finální JSON nebo posílat validní Server-Sent Events (SSE).

---

## 💻 Požadovaná struktura projektu

Projekt uspořádej následovně:
```text
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI aplikace, endpoint POST /ask
│   ├── config.py        # Načítání proměnných z .env přes Pydantic-Settings
│   ├── database.py      # Inicializace ChromaDB, naplnění (ingest) dat
│   ├── services.py      # Logika RAG (embedding, vyhledávání, LLM prompt)
│   └── models.py        # Pydantic schémata pro API (Request/Response)
├── data/
│   └── transcript.txt   # Zdrojový text dokumentu
├── DESIGN.md            # Dokumentace návrhu (chunking, retrieval, prompts)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 📋 Specifikace Implementace Krok za Krokem

### Krok 1: Datové modely (`app/models.py`)
Definuj přesnou strukturu odpovědi podle zadání:
```python
from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str

class SourceReference(BaseModel):
    timestamp: str  # Formát HH:MM:SS nebo MM:SS vytažený z metadat
    excerpt: str    # Krátký výsek textu, ze kterého se čerpalo

class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
```

### Krok 2: Ingest proces (`app/database.py`)
Při startu aplikace (nebo v samostatném kroku v Dockeru před spuštěním API) zkontroluj, zda databáze existuje. Pokud ne:
1. Načti `data/transcript.txt`.
2. Projdi řádky, extrahuj časové značky.
3. Rozsekej text na chunky.
4. Vygeneruj embeddingy a ulož dokumenty s metadaty (`{"timestamp": current_timestamp, "excerpt": chunk_text}`) do ChromaDB kolekce.

### Krok 3: Vyhledávání a Prompt Engineering (`app/services.py`)
Při dotazu na `POST /ask`:
1. Převeď dotaz na embedding vektor pomocí zvoleného poskytovatele (Gemini / Ollama).
2. Vyhledej top 3 nejbližší chunky v ChromaDB (použij kosinovou podobnost).
3. Sestav prompt pro LLM:
   ```text
   Context information is below.
   ---------------------
   {context_str}
   ---------------------
   Given the context information and not prior knowledge, answer the query.
   Query: {query_str}
   
   Strict Rules:
   1. Answer the query in natural language based SOLELY on the context provided.
   2. If the context does not contain the answer, reply exactly with: "I am sorry, but the provided material does not contain an answer to this question." Do not attempt to fabricate an answer.
   ```

### Krok 4: FastAPI Router (`app/main.py`)
Vytvoř asynchronní endpoint, který přijme `QuestionRequest`, zavolá vyhledávací pipeline a vrátí `QuestionResponse`. Ošetři try-except bloky pro případ výpadku externího API, aby server neposlal 500 Internal Server Error bez vysvětlení, ale vrátil elegantní chybovou hlášku.

### Krok 5: Docker konfigurace (`docker-compose.yml`)
Zajisti, aby kontejner s FastAPI aplikací správně mapoval port `8000` na hostitele a automaticky četl proměnné prostředí z `.env`.

---

## 🏁 Akceptační testy (Na co si dát pozor při kontrole)
Před odevzdáním musí kód bezchybně projít těmito scénáři:
1. **Fakta:** Otázka "Jak dopadl pokus s boraxem v mléku?" musí vrátit správnou odpověď a zdroje s časem kolem `00:10:33`.
2. **Syntéza:** Otázka porovnávající Viktoriánskou a Edwardskou domácnost musí složit fakta z obou částí dokumentu.
3. **Neexistence:** Otázka "Jaká nebezpečí hrozila v kuchyních starověkého Říma?" musí vrátit definovanou hlášku o tom, že informace v textu nejsou, a pole `sources` musí být prázdné nebo obsahovat pouze nerelevantní pokusy o shodu s nulovým skóre (ideálně prázdné).

Pusť se do generování čistého, modulárního kódu s typovými anotacemi (type hinting) a popisy funkcí (docstrings).