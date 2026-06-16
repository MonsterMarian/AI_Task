import os
import sys
import time
import logging

# Set up simple stdout logging to avoid cluttering verification output
logging.basicConfig(level=logging.ERROR)

# Ensure app directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.database import get_chroma_client, ingest_data
from app.services import answer_question

# ANSI Color codes for nice CLI layout
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_separator():
    print("-" * 80)

def main():
    print(f"{Colors.BOLD}{Colors.HEADER}=== RAG ASSISTANT VERIFICATION SUITE ==={Colors.ENDC}\n")

    # 1. Check/Initialize Database
    print(f"{Colors.BOLD}[1/3] Database Initialization...{Colors.ENDC}")
    start_time = time.time()
    try:
        client = get_chroma_client()
        ingest_data(client)
        print(f"{Colors.GREEN}[OK] Database is populated.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}[FAIL] Database initialization failed: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Please make sure GEMINI_API_KEY is set in '.env' or Ollama is running.{Colors.ENDC}")
        sys.exit(1)
    print_separator()

    # 2. Run Test Cases
    print(f"{Colors.BOLD}[2/3] Executing 5 Required Scenarios...{Colors.ENDC}\n")
    
    test_cases = [
        {
            "name": "Scenario 1: Factual Question (Fakta)",
            "query": "Jak dopadl pokus s boraxem v mléku?",
            "validator": lambda ans, sources: any("00:10:33" in s["timestamp"] for s in sources) or any("00:11:30" in s["timestamp"] for s in sources),
            "validation_msg": "Should return sources with timestamp around 00:10:33 or 00:11:30."
        },
        {
            "name": "Scenario 2: Synthesis Question (Syntéza)",
            "query": "Porovnej nebezpečí a rizika ve viktoriánské a edwardské domácnosti.",
            "validator": lambda ans, sources: any(s["timestamp"] < "00:58:00" for s in sources) and any(s["timestamp"] >= "00:58:00" for s in sources),
            "validation_msg": "Should retrieve sources from both Victorian (< 00:58:00) and Edwardian (>= 00:58:00) periods."
        },
        {
            "name": "Scenario 3: Named Person/Location (Vynálezci)",
            "query": "Kdo byl Thomas Crapper a jaký vynález zachránil lidi před výbuchem?",
            "validator": lambda ans, sources: "crapper" in ans.lower() or "sifon" in ans.lower() or "ventil" in ans.lower() or "methan" in ans.lower(),
            "validation_msg": "Should mention Thomas Crapper, the siphon valve / water trap, or methane."
        },
        {
            "name": "Scenario 4: Vague/Broad Question (Široká)",
            "query": "Jak lidé v historii bojovali s nečistotou a jaké to mělo následky?",
            "validator": lambda ans, sources: len(ans) > 50,
            "validation_msg": "Should answer in natural language based on transcript history (washing, laundry, mangles, gas baths)."
        },
        {
            "name": "Scenario 5: Out of Scope Question (Neexistence)",
            "query": "Jaká nebezpečí hrozila v kuchyních starověkého Říma?",
            "validator": lambda ans, sources: ans == "I am sorry, but the provided material does not contain an answer to this question." and len(sources) == 0,
            "validation_msg": "Should reply exactly with: 'I am sorry...' and return empty sources."
        }
    ]

    passed_count = 0

    for idx, tc in enumerate(test_cases, start=1):
        print(f"{Colors.BOLD}{Colors.BLUE}Test #{idx}: {tc['name']}{Colors.ENDC}")
        print(f"{Colors.CYAN}Query: {tc['query']}{Colors.ENDC}")
        
        t0 = time.time()
        try:
            answer, sources = answer_question(tc["query"])
            duration = time.time() - t0
            
            print(f"{Colors.GREEN}Answer: {answer}{Colors.ENDC}")
            print(f"Sources: {[s['timestamp'] for s in sources]}")
            print(f"Duration: {duration:.2f}s")
            
            # Run validator
            is_valid = tc["validator"](answer, sources)
            if is_valid:
                print(f"{Colors.GREEN}[OK] VALIDATED{Colors.ENDC}")
                passed_count += 1
            else:
                print(f"{Colors.WARNING}[WARN] VALIDATION WARNING: {tc['validation_msg']}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}[FAIL] ERROR running query: {e}{Colors.ENDC}")
            
        print_separator()

    # 3. Report
    print(f"{Colors.BOLD}[3/3] Final Report{Colors.ENDC}")
    print(f"Passed: {passed_count}/{len(test_cases)}")
    if passed_count == len(test_cases):
        print(f"{Colors.GREEN}{Colors.BOLD}ALL SCENARIOS COMPLETED SUCCESSFULLY! [OK]{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}Completed with some warnings. Please check details above. [WARN]{Colors.ENDC}")

if __name__ == "__main__":
    main()
