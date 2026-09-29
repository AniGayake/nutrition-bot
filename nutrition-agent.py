import os
import sqlite3
import asyncio
import warnings
from datetime import date
from typing import List

from aiohttp import web
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field

# Silence non-critical thought signature warnings from the SDK
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

# ---------------------------------------------------------
# 1. Database Initialization (SQLite)
# ---------------------------------------------------------
DB_FILE = "nutrition.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
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

# ---------------------------------------------------------
# 2. Pydantic Models for Agent Tool
# ---------------------------------------------------------
class FoodItem(BaseModel):
    food_name: str = Field(description="Name of the food item")
    portion: str = Field(description="Portion size consumed")
    calories: float = Field(description="Estimated calories in kcal")
    protein_g: float = Field(description="Estimated protein in grams")
    carbs_g: float = Field(description="Estimated carbohydrates in grams")
    fat_g: float = Field(description="Estimated fat in grams")

# ---------------------------------------------------------
# 3. Gemini Client & Scoped Execution
# ---------------------------------------------------------
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def run_nutrition_agent(user_id: int, user_message: str) -> str:
    """Executes the nutrition agent with user-scoped database tools."""

    def log_food_items(items: List[FoodItem], log_date: str = "") -> dict:
        """Logs one or more food items into the database for this specific user."""
        try:
            target_date = log_date.strip() if log_date else str(date.today())
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            saved = []
            for item in items:
                cursor.execute("""
                    INSERT INTO food_logs (user_id, log_date, food_name, portion, calories, protein_g, carbs_g, fat_g)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    target_date,
                    item.food_name,
                    item.portion,
                    item.calories,
                    item.protein_g,
                    item.carbs_g,
                    item.fat_g
                ))
                saved.append(f"{item.food_name} ({item.portion})")
            conn.commit()
            conn.close()
            return {"status": "success", "date": target_date, "logged_items": saved}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def get_daily_nutrition(log_date: str = "") -> dict:
        """Retrieves all logged foods and macro totals for this user on a specified date."""
        try:
            target_date = log_date.strip() if log_date else str(date.today())
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT food_name, portion, calories, protein_g, carbs_g, fat_g
                FROM food_logs
                WHERE user_id = ? AND log_date = ?
            """, (user_id, target_date))
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {
                    "status": "success",
                    "date": target_date,
                    "has_logs": False,
                    "items": [],
                    "totals": {
                        "total_calories": 0.0,
                        "total_protein_g": 0.0,
                        "total_carbs_g": 0.0,
                        "total_fat_g": 0.0
                    }
                }

            items = []
            total_cal = total_p = total_c = total_f = 0.0
            for name, portion, cal, p, c, f in rows:
                items.append({
                    "food": name,
                    "portion": portion,
                    "calories": cal,
                    "protein": p,
                    "carbs": c,
                    "fat": f
                })
                total_cal += (cal or 0)
                total_p += (p or 0)
                total_c += (c or 0)
                total_f += (f or 0)

            return {
                "status": "success",
                "date": target_date,
                "has_logs": True,
                "items": items,
                "totals": {
                    "total_calories": round(total_cal, 1),
                    "total_protein_g": round(total_p, 1),
                    "total_carbs_g": round(total_c, 1),
                    "total_fat_g": round(total_f, 1)
                }
            }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    config = types.GenerateContentConfig(
        system_instruction=(
            f"You are a personal nutrition assistant. Today's date is {date.today()}.\n"
            "When the user mentions what they ate or drank, estimate calories, protein (g), carbs (g), and fat (g), "
            "then invoke `log_food_items`.\n"
            "When the user asks what they ate, asks for progress, or requests daily totals, invoke `get_daily_nutrition`.\n"
            "IMPORTANT: If `get_daily_nutrition` returns `has_logs: false`, do NOT say there is a technical problem. "
            "Simply state that no meals have been logged yet for that date and invite them to log their first meal.\n"
            "Keep replies clean, structured, and encouraging."
        ),
        tools=[log_food_items, get_daily_nutrition],
    )

    # Multi-model failover for free-tier resilience
    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

    for model_name in models_to_try:
        try:
            chat = gemini_client.chats.create(model=model_name, config=config)
            response = chat.send_message(user_message)
            return response.text
        except errors.ServerError as e:
            if "503" in str(e):
                continue
            return "Google API is temporarily busy. Please try again in a few seconds."
        except Exception as e:
            return f"Error processing meal log: {e}"

    return "All model endpoints are currently experiencing high demand. Please try again in 1–2 minutes."

# ---------------------------------------------------------
# 4. Telegram Message Handler
# ---------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Run blocking Gemini + SQLite processing in an executor pool to keep the event loop responsive
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, run_nutrition_agent, user_id, user_text)

    await update.message.reply_text(reply)

# ---------------------------------------------------------
# 5. Render HTTP Server & Main Polling Runner
# ---------------------------------------------------------
async def health_check(request):
    """Answers Render's port binding check and external uptime pingers."""
    return web.Response(text="Nutrition Agent Bot is live and healthy!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()

    # Render injects the PORT environment variable dynamically (defaults to 8080 locally)
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Health check server listening on port {port}")

async def main():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not bot_token:
        raise ValueError("Missing required environment variable: TELEGRAM_BOT_TOKEN")
    if not gemini_key:
        raise ValueError("Missing required environment variable: GEMINI_API_KEY")

    # Start dummy web server for Render health checks
    await start_web_server()

    # Initialize Telegram polling bot
    telegram_app = ApplicationBuilder().token(bot_token).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Starting Telegram long polling...")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)

    # Keep the asyncio event loop alive
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())