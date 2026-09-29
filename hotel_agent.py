import os
import requests
import asyncio
import aiohttp
import warnings
from datetime import date
from typing import Optional

from aiohttp import web
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google import genai
from google.genai import types, errors

warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

# 1. Scrape.do Tool Function
def search_hotel_deals(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    max_price_inr: Optional[float] = None,
    min_rating: Optional[float] = None
) -> dict:
    token = os.environ.get("SCRAPEDO_TOKEN")
    if not token:
        return {"error": "SCRAPEDO_TOKEN is missing."}

    url = "https://api.scrape.do/plugin/google/hotels"
    params = {
        "token": token,
        "q": f"{location} hotels",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "currency": "INR",
        "gl": "in",
        "hl": "en",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            return {"error": f"Scrape.do HTTP error {response.status_code}: {response.text[:200]}"}

        data = response.json()
        properties = data.get("properties") or data.get("hotels") or (data if isinstance(data, list) else [])

        if not properties:
            return {"status": "no_results", "message": f"No hotels returned for {location}."}

        curated_deals = []
        for prop in properties:
            name = prop.get("name") or prop.get("hotel_name")
            rating = prop.get("rating") or prop.get("overall_rating", 0.0)
            reviews = prop.get("reviews") or prop.get("review_count", 0)

            try:
                numeric_rating = float(rating) if rating else 0.0
            except ValueError:
                numeric_rating = 0.0

            if min_rating and numeric_rating < min_rating:
                continue

            price_str = (
                prop.get("price")
                or prop.get("rate_per_night", {}).get("lowest")
                or prop.get("lowest_price")
            )

            numeric_price = None
            if price_str:
                clean_num = ''.join(c for c in str(price_str) if c.isdigit())
                numeric_price = float(clean_num) if clean_num else None

            if max_price_inr and numeric_price and numeric_price > max_price_inr:
                continue

            prices_breakdown = []
            ota_deals = prop.get("prices") or prop.get("booking_providers", [])
            for deal in ota_deals:
                source = deal.get("source") or deal.get("provider") or "Direct"
                deal_price = deal.get("rate_per_night", {}).get("lowest") or deal.get("price")
                deal_link = deal.get("link") or deal.get("booking_link")
                prices_breakdown.append({
                    "provider": source,
                    "price": deal_price,
                    "link": deal_link
                })

            curated_deals.append({
                "hotel_name": name,
                "rating": numeric_rating,
                "reviews": reviews,
                "price_per_night": price_str or "Check site",
                "direct_link": prop.get("link") or prop.get("property_token_link"),
                "comparisons": prices_breakdown[:3]
            })

        return {
            "status": "success",
            "destination": location,
            "dates": f"{check_in_date} to {check_out_date}",
            "matches_found": len(curated_deals),
            "hotels": curated_deals[:6]
        }
    except Exception as e:
        return {"error": f"Failed fetching hotel deals: {str(e)}"}

# 2. Gemini Agent
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def run_agent_query(user_query: str) -> str:
    today_str = str(date.today())
    chat_config = types.GenerateContentConfig(
        system_instruction=(
            f"You are a hotel deal aggregator agent.\n"
            f"CRITICAL DATE CONTEXT: Today's current date is {today_str}.\n"
            "All travel dates must be future dates relative to today. Never query dates in the past.\n"
            "When the user requests stays, invoke the `search_hotel_deals` tool.\n"
            "Present results in a clean Markdown comparison table: "
            "Hotel Name | Rating | Lowest Price | Provider.\n"
            "Include links and conclude with a quick 1-line recommendation of the best deal."
        ),
        tools=[search_hotel_deals],
    )

    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    for model_name in models_to_try:
        try:
            chat = gemini_client.chats.create(model=model_name, config=chat_config)
            response = chat.send_message(user_query)
            return response.text
        except errors.ServerError:
            continue
        except Exception as e:
            return f"Error: {e}"

    return "All model endpoints are busy. Please try again in 1 minute."

# 3. Telegram Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, run_agent_query, user_text)
    await update.message.reply_text(reply)

# 4. HTTP Health Check & Self-Pinger for Render
async def health_check(request):
    return web.Response(text="Hotel Deal Agent is live!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Health server listening on port {port}")

async def keep_alive_ping():
    """Pings itself every 10 minutes to stay awake on Render free tier."""
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not render_url:
        return
    if not render_url.startswith("http"):
        render_url = f"https://{render_url}"

    await asyncio.sleep(60)
    print(f"[Keep-Alive] Starting pinger on {render_url}")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(render_url, timeout=15) as resp:
                    print(f"[Keep-Alive] Ping: Status {resp.status}")
            except Exception as e:
                print(f"[Keep-Alive Ping Failed]: {e}")
            await asyncio.sleep(600)

# 5. Main Entrypoint (NO input() calls!)
async def main():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN")

    await start_web_server()
    asyncio.create_task(keep_alive_ping())

    telegram_app = ApplicationBuilder().token(bot_token).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Hotel Telegram Agent is polling...")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)

    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())