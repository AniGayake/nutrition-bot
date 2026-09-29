import os
import requests
import warnings
from datetime import date
from typing import Optional
from google import genai
from google.genai import types

# Silence thought_signature SDK notice
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

# -------------------------------------------------------------------
# 1. Scrape.do Google Hotels Integration
# -------------------------------------------------------------------
def search_hotel_deals(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    max_price_inr: Optional[float] = None,
    min_rating: Optional[float] = None
) -> dict:
    """Scrapes hotel booking sites via Scrape.do Google Hotels plugin."""
    print(f"\n[Scrape.do Executing] Searching {location} ({check_in_date} to {check_out_date})...")

    token = os.environ.get("SCRAPEDO_TOKEN")
    if not token:
        print("[Error] SCRAPEDO_TOKEN is not set in environment!")
        return {"error": "SCRAPEDO_TOKEN is missing."}

    # Scrape.do dedicated Google Hotels Ready-API endpoint
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
            print(f"[Scrape.do HTTP Error]: {response.status_code} - {response.text}")
            return {"error": f"Scrape.do responded with code {response.status_code}: {response.text[:200]}"}

        data = response.json()

        # Handle different response wrappers if returned as a list or dict
        properties = data.get("properties") or data.get("hotels") or (data if isinstance(data, list) else [])

        if not properties:
            print(f"[Scrape.do Info]: No properties in payload. Keys: {list(data.keys()) if isinstance(data, dict) else 'List'}")
            return {"status": "no_results", "message": f"No hotels returned for {location}."}

        print(f"[Scrape.do Success]: Retrieved {len(properties)} properties. Filtering best deals...")

        curated_deals = []
        for prop in properties:
            name = prop.get("name") or prop.get("hotel_name")
            rating = prop.get("rating") or prop.get("overall_rating", 0.0)
            reviews = prop.get("reviews") or prop.get("review_count", 0)

            # Rating filter
            try:
                numeric_rating = float(rating) if rating else 0.0
            except ValueError:
                numeric_rating = 0.0

            if min_rating and numeric_rating < min_rating:
                continue

            # Price extraction
            price_str = (
                prop.get("price")
                or prop.get("rate_per_night", {}).get("lowest")
                or prop.get("lowest_price")
            )

            numeric_price = None
            if price_str:
                clean_num = ''.join(c for c in str(price_str) if c.isdigit())
                numeric_price = float(clean_num) if clean_num else None

            # Budget filter
            if max_price_inr and numeric_price and numeric_price > max_price_inr:
                continue

            # OTA booking comparisons
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
            "hotels": curated_deals[:7]
        }

    except Exception as e:
        print(f"[Exception during scrape.do call]: {e}")
        return {"error": f"Failed fetching hotel deals: {str(e)}"}

# -------------------------------------------------------------------
# 2. Interactive Gemini Chat Runner
# -------------------------------------------------------------------
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please export GEMINI_API_KEY.")

    client = genai.Client(api_key=api_key)

    today_str = str(date.today())

    chat_config = types.GenerateContentConfig(
        system_instruction=(
            f"You are a hotel deal aggregator agent.\n"
            f"CRITICAL DATE CONTEXT: Today's current date is {today_str}.\n"
            "All travel dates must be future dates relative to today. Never query dates in 2024 or earlier.\n"
            "When the user requests stays, invoke the `search_hotel_deals` tool.\n"
            "Present results in a clean Markdown comparison table: "
            "Hotel Name | Rating | Lowest Price | Provider.\n"
            "Include links and conclude with a quick 1-line recommendation of the best deal."
        ),
        tools=[search_hotel_deals],
    )

    chat = client.chats.create(model="gemini-3.5-flash-lite", config=chat_config)

    print("=" * 60)
    print(f"🏨 Scrape.do Hotel Deal Agent Active! (Today: {today_str})")
    print("Example: 'Find hotels in Goa for next weekend under 4500 INR'")
    print("Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            response = chat.send_message(user_input)
            print(f"\nAgent:\n{response.text}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession stopped.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    main()