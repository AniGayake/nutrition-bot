import os
import sqlite3
import time
from datetime import date
from typing import List
from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field

DB_FILE = "nutrition.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date TEXT NOT NULL,
            food_name TEXT NOT NULL,
            portion TEXT,
            calories REAL,
            protein_g REAL,
            carbs_g REAL,
            fat_g REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class FoodItem(BaseModel):
    food_name: str = Field(description="Food item name")
    portion: str = Field(description="Portion size consumed")
    calories: float = Field(description="Estimated calories in kcal")
    protein_g: float = Field(description="Estimated protein in grams")
    carbs_g: float = Field(description="Estimated carbohydrates in grams")
    fat_g: float = Field(description="Estimated fat in grams")

def log_food_items(items: List[FoodItem], log_date: str = "") -> dict:
    """Logs one or more food items into the database."""
    target_date = log_date if log_date else str(date.today())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    saved = []
    for item in items:
        cursor.execute("""
            INSERT INTO food_logs (log_date, food_name, portion, calories, protein_g, carbs_g, fat_g)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (target_date, item.food_name, item.portion, item.calories, item.protein_g, item.carbs_g, item.fat_g))
        saved.append(f"{item.food_name} ({item.portion})")
    conn.commit()
    conn.close()
    return {"status": "success", "date": target_date, "logged_items": saved}

def get_daily_nutrition(log_date: str = "") -> dict:
    """Retrieves all logged foods and aggregate totals for a given date."""
    target_date = log_date if log_date else str(date.today())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT food_name, portion, calories, protein_g, carbs_g, fat_g FROM food_logs WHERE log_date = ?", (target_date,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {"date": target_date, "message": "No meals logged for this date."}

    items = []
    total_cal = total_p = total_c = total_f = 0.0
    for name, portion, cal, p, c, f in rows:
        items.append({"food": name, "portion": portion, "calories": cal, "protein": p, "carbs": c, "fat": f})
        total_cal += (cal or 0)
        total_p += (p or 0)
        total_c += (c or 0)
        total_f += (f or 0)

    return {
        "date": target_date,
        "items": items,
        "totals": {
            "total_calories": round(total_cal, 1),
            "total_protein_g": round(total_p, 1),
            "total_carbs_g": round(total_c, 1),
            "total_fat_g": round(total_f, 1)
        }
    }

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

chat_config = types.GenerateContentConfig(
    system_instruction=(
        f"You are a personal nutrition assistant. Today's date is {date.today()}.\n"
        "When a user tells you what they ate, estimate the calories, protein, carbs, and fat "
        "and invoke `log_food_items`.\n"
        "When they ask what they ate or for a summary, invoke `get_daily_nutrition`.\n"
        "Provide concise, encouraging feedback with a clean macro table."
    ),
    tools=[log_food_items, get_daily_nutrition],
)

# Ordered list of models to try (Flash-Lite first for low latency & reliability)
AVAILABLE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

def send_message_with_fallback(prompt: str) -> str:
    for model_name in AVAILABLE_MODELS:
        try:
            # Create a session on the available model
            chat_session = client.chats.create(model=model_name, config=chat_config)
            return chat_session.send_message(prompt).text
        except errors.ServerError as e:
            if "503" in str(e):
                print(f"[{model_name} is under high demand (503). Switching to fallback...]")
                continue
            raise e
    raise RuntimeError("All available model endpoints are temporarily experiencing high demand. Please try again in 1-2 minutes.")

if __name__ == "__main__":
    print(f"Nutrition Tracker Agent Ready! (Date: {date.today()})\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        try:
            reply = send_message_with_fallback(user_input)
            print(f"\nAgent: {reply}\n")
        except Exception as err:
            print(f"\n[Error]: {err}\n")