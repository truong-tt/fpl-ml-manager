import os
import subprocess
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load variables from .env
load_dotenv()

def extract_code(text: str) -> str:
    """Extracts python code from Gemini's markdown response."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def run_verifier() -> float | None:
    """Runs train.py and extracts the MAE score, with a 5-minute timeout."""
    try:
        result = subprocess.run(
            ["python", "train.py"], 
            capture_output=True, 
            text=True, 
            timeout=300 
        )
        match = re.search(r"VERIFIER_SCORE:\s*([0-9.]+)", result.stdout)
        if match:
            return float(match.group(1))
        
        print("Execution failed or no score found:\n", result.stderr)
        return None

    except subprocess.TimeoutExpired:
        print("Execution timed out after 5 minutes!")
        return None

def main():
    with open("program.md", "r") as f:
        system_instruction = f.read()

    # Fetch key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )

    with open("train.py", "r") as f:
        current_code = f.read()
        
    current_best_score = run_verifier()
    if current_best_score is None:
        print("Baseline backtest failed. Fix train.py first.")
        return

    print(f"BASELINE SCORE: {current_best_score}")
    
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=config
    )
    
    prompt = f"""
    Here is the baseline `train.py` code. The current baseline MAE score is {current_best_score}.
    Propose an improvement to lower the MAE, and output the complete, updated `train.py` script.
    ```python\n{current_code}\n```
    """

    for i in range(50):
        print(f"\n--- Iteration {i+1} ---")
        
        response = chat.send_message(prompt)
        new_code = extract_code(response.text)
        
        with open("train.py", "w") as f:
            f.write(new_code)
            
        new_score = run_verifier()
        
        if new_score is None:
            prompt = "The code crashed or failed. Fix the bug and output the full corrected code."
            os.system("git checkout train.py")
        elif new_score < current_best_score:
            print(f"NEW SOTA! Score improved: {current_best_score} -> {new_score}")
            current_best_score = new_score
            os.system("git add train.py && git commit -m 'Auto-Research: new features'")
            prompt = f"Success! Score is now {new_score}. Keep this as baseline."
        else:
            print(f"Failed. Score worsened to {new_score}. Reverting...")
            os.system("git checkout train.py")
            prompt = f"Failed. Score worsened to {new_score}. Try a different approach."
            
        print("Waiting 15 seconds to respect API rate limits...")
        time.sleep(15) 

if __name__ == "__main__":
    main()