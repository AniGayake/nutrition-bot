import os
import requests
import asyncio
import aiohttp
import warnings
from datetime import date
from typing import Optional

from aiohttp import web
from telegram import Update
from telegram.request import HTTPXRequest
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google import genai
from google.genai import types, errors

# Silence non-critical thought signature notices from the Google GenAI SDK
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

# -------------------------------------------------------------------
# 1. Scrape.do Google Hotels Integration Tool
# -------------------------------------------------------------------
def search_hotel_deals(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    max_price_inr: Optional[float] = None,
    min_rating: Optional[float] = None
) -> dict:
    """Scrapes hotel booking websites via Scrape.do Google Hotels plugin.

    Args:
        location: Target city or area (e.g., 'Goa', 'Manali', 'Jaipur').
        check_in_date: Check-in date in YYYY-MM-DD format.
        check_out_date: Check-out date in YYYY-MM-DD format.
        adults: Total adult guests (default 2).
        max_price_inr: Maximum budget ceiling in INR per night.
        min_rating: Minimum guest review rating (e.g., 4.0).
    """
    token = os.environ.get("SCRAPEDO_TOKEN", "").strip().strip("'").strip('"')
    if not token:
        return {"status": "error", "message": "SCRAPEDO_TOKEN is not set in environment."}

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
            return {
                "status": "error",
                "message": f"Scrape.do responded with code {response.status_code}: {response.text[:200]}"
            }

        data = response.json()
        properties = data.get("properties") or data.get("hotels") or (data if isinstance(data, list) else [])

        if not properties:
            return {
                "status": "no_results",
                "message": f"No available properties returned for {location} from {check_in_date} to {check_out_date}."
            }

        curated_deals = []
        for prop in properties:
            name = prop.get("name") or prop.get("hotel_name")
            rating = prop.get("rating") or prop.get("overall_rating", 0.0)
            reviews = prop.get("reviews") or prop.get("review_count", 0)

            try:
                numeric_rating = float(rating) if rating else 0.0
            except ValueError:
                numeric_rating = 0.0

            # Filter by review score
            if min_rating and numeric_rating < min_rating:
                continue

            price_str = (
                prop.get("price")
                or prop.get("rate_per_night", {}).get("lowest")
                or prop.get("lowest_price")
            )

            numeric_price = None
            if price_str:
                clean_num = "".join(c for c in str(price_str) if c.isdigit())
                numeric_price = float(clean_num) if clean_num else None

            # Filter by price budget
            if max_price_inr and numeric_price and numeric_price > max_price_inr:
                continue

            # Parse competitor OTA price breakdowns (Booking.com, Agoda, Expedia, etc.)
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
                "lowest_price": price_str or "Check site",
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
        return {"status": "error", "message": f"Scraper execution error: {str(e)}"}

# -------------------------------------------------------------------
# 2. Gemini Agent Scoped Runner
# -------------------------------------------------------------------
gemini_key = os.environ.get("GEMINI_API_KEY", "").strip().strip("'").strip('"')
gemini_client = genai.Client(api_key=gemini_key)

def run_hotel_agent(user_query: str) -> str:
    """Invokes Gemini with tools and returns clean natural-language advice."""
    today_str = str(date.today())

    chat_config = types.GenerateContentConfig(
        system_instruction=(
            f"You are an expert hotel deal hunter and aggregator bot.\n"
            f"IMPORTANT: Today's current date is {today_str}.\n"
            "All travel dates MUST be future dates relative to today. Never query dates in 2024 or earlier.\n"
            "When the user requests stays, deals, or bookings, extract location, check-in, check-out dates, "
            "budget, and rating requirements, and invoke `search_hotel_deals`.\n"
            "If the user didn't specify exact dates, assume a 2-night weekend stay starting next Friday.\n\n"
            "When formatting the output for Telegram:\n"
            "1. Present results with clean formatting: Hotel Name, Star/Review Rating, and Lowest Price.\n"
            "2. Detail which provider (Agoda, Booking.com, Official) offers the best rate.\n"
            "3. Include direct booking links where available.\n"
            "4. Conclude with a 1-line recommendation of the best value-for-money option."
        ),
        tools=[search_hotel_deals],
    )

    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

    for model_name in models_to_try:
        try:
            chat = gemini_client.chats.create(model=model_name, config=chat_config)
            response = chat.send_message(user_query)
            return response.text
        except errors.ServerError as e:
            if "503" in str(e):
                continue
            return "Google API is temporarily busy. Please try your request again shortly."
        except Exception as e:
            return f"Error processing hotel search: {e}"

    return "All model clusters are temporarily busy. Please retry in 1–2 minutes."

# -------------------------------------------------------------------
# 3. Telegram Message Handler
# -------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Run blocking Gemini and Scrape.do calls in an executor pool to keep the event loop alive
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, run_hotel_agent, user_text)

    await update.message.reply_text(reply)

# -------------------------------------------------------------------
# 4. HTTP Health Server & Render Auto-Pinger
# -------------------------------------------------------------------
async def health_check(request):
    """Responds to Render port binding check and external pingers."""
    return web.Response(text="Hotel Deal Agent is live and healthy!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()

    # Render provides PORT dynamically (defaults to 8080 locally)
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Health server listening on port {port}")

async def keep_alive_ping():
    """Periodically pings public Render URL every 10 minutes to prevent container sleep."""
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not render_url:
        return
    if not render_url.startswith("http"):
        render_url = f"https://{render_url}"

    await asyncio.sleep(60)  # Wait 1 minute after boot
    print(f"[Keep-Alive] Pinger active for: {render_url}")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(render_url, timeout=15) as resp:
                    print(f"[Keep-Alive] Pinged server: Status {resp.status}")
            except Exception as e:
                print(f"[Keep-Alive] Ping failed: {e}")
            await asyncio.sleep(600)  # 10 minutes

# -------------------------------------------------------------------
# 5. Main Application Entrypoint
# -------------------------------------------------------------------
async def main():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip().strip("'").strip('"')
    if not bot_token:
        raise ValueError("Missing required environment variable: TELEGRAM_BOT_TOKEN")

    # Start HTTP dummy web server for Render health checks
    await start_web_server()

    # Start keep-alive loop
    asyncio.create_task(keep_alive_ping())

    # Configure robust network timeouts (30s) to prevent httpx.ReadTimeout during handshake
    t_request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0
    )

    telegram_app = (
        ApplicationBuilder()
        .token(bot_token)
        .request(t_request)
        .get_updates_request(t_request)
        .build()
    )

    telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Hotel Telegram Agent is polling...")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)

    # Keep asyncio loop alive indefinitely
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())