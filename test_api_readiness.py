#!/usr/bin/env python3
"""Test script to verify all API connections before going live."""

import os
import sys
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import requests
import asyncio


def test_alpaca_rest():
    """Test Alpaca REST API - account info."""
    print("\n1. ALPACA REST API (Account Info)")
    print("-" * 40)

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("✗ MISSING: ALPACA_API_KEY or ALPACA_SECRET_KEY not set")
        return False

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    url = "https://paper-api.alpaca.markets/v2/account"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ SUCCESS - Account Status: {data.get('status')}")
            print(f"  Buying Power: ${float(data.get('buying_power', 0)):,.2f}")
            print(f"  Cash: ${float(data.get('cash', 0)):,.2f}")
            print(f"  Portfolio Value: ${float(data.get('portfolio_value', 0)):,.2f}")
            return True
        else:
            print(f"✗ FAILED - Status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_alpaca_options_chain():
    """Test Alpaca Options REST API - get option chain."""
    print("\n2. ALPACA OPTIONS REST API (Option Chain)")
    print("-" * 40)

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    # Get option contracts for NVDA
    url = "https://paper-api.alpaca.markets/v2/options/contracts"
    params = {
        "underlying_symbols": "NVDA",
        "status": "active",
        "limit": 5,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            contracts = data.get("option_contracts", [])
            print(f"✓ SUCCESS - Found {len(contracts)} option contracts")
            if contracts:
                c = contracts[0]
                print(f"  Sample: {c.get('symbol')} - {c.get('strike_price')} {c.get('type')}")
            return True
        else:
            print(f"✗ FAILED - Status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


async def test_alpaca_websocket():
    """Test Alpaca WebSocket connection (may fail outside market hours)."""
    print("\n3. ALPACA WEBSOCKET (Options Stream)")
    print("-" * 40)

    import websockets
    import json

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    url = "wss://stream.data.alpaca.markets/v1beta1/opra"

    try:
        async with websockets.connect(url, close_timeout=5) as ws:
            # Send auth
            auth_msg = {
                "action": "auth",
                "key": api_key,
                "secret": secret_key,
            }
            await ws.send(json.dumps(auth_msg))

            # Wait for response
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(response)

            if isinstance(data, list) and len(data) > 0:
                msg = data[0]
                if msg.get("T") == "success":
                    print(f"✓ SUCCESS - WebSocket connected and authenticated")
                    print(f"  Message: {msg.get('msg', 'authenticated')}")
                    return True
                else:
                    print(f"✓ CONNECTED - Response: {msg}")
                    return True
            else:
                print(f"? RESPONSE: {data}")
                return True

    except asyncio.TimeoutError:
        print("✗ TIMEOUT - No response (may be outside market hours)")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_orats():
    """Test ORATS API - get Greeks data."""
    print("\n4. ORATS API (Greeks)")
    print("-" * 40)

    api_token = os.getenv("ORATS_API_TOKEN")

    if not api_token:
        print("✗ MISSING: ORATS_API_TOKEN not set")
        return False

    url = "https://api.orats.io/datav2/strikes"
    params = {
        "token": api_token,
        "ticker": "NVDA",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            strikes = data.get("data", [])
            print(f"✓ SUCCESS - Got {len(strikes)} strikes for NVDA")
            if strikes:
                s = strikes[0]
                print(f"  Sample: ${s.get('strike')} {s.get('expirDate')} - Delta: {s.get('callDelta', 'N/A')}")
            return True
        else:
            print(f"✗ FAILED - Status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_finnhub():
    """Test Finnhub API - get news sentiment."""
    print("\n5. FINNHUB API (News Sentiment)")
    print("-" * 40)

    api_key = os.getenv("FINNHUB_API_KEY")

    if not api_key:
        print("✗ MISSING: FINNHUB_API_KEY not set")
        return False

    url = "https://finnhub.io/api/v1/news-sentiment"
    params = {
        "symbol": "NVDA",
        "token": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            sentiment = data.get("sentiment", {})
            buzz = data.get("buzz", {})
            print(f"✓ SUCCESS - Got sentiment for NVDA")
            print(f"  Bullish: {sentiment.get('bullishPercent', 'N/A')}")
            print(f"  Bearish: {sentiment.get('bearishPercent', 'N/A')}")
            print(f"  Articles: {buzz.get('articlesInLastWeek', 'N/A')}")
            return True
        elif resp.status_code == 403:
            print(f"✗ RATE LIMITED or UNAUTHORIZED - May need premium plan")
            return False
        else:
            print(f"✗ FAILED - Status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_quiver():
    """Test Quiver API - get WSB sentiment."""
    print("\n6. QUIVER API (WSB Sentiment)")
    print("-" * 40)

    api_key = os.getenv("QUIVER_API_KEY")

    if not api_key:
        print("✗ MISSING: QUIVER_API_KEY not set")
        return False

    url = "https://api.quiverquant.com/beta/live/wallstreetbets"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ SUCCESS - Got WSB data, {len(data)} entries")
            # Find NVDA in the data
            nvda = next((d for d in data if d.get("Ticker") == "NVDA"), None)
            if nvda:
                print(f"  NVDA: Mentions={nvda.get('Mentions', 'N/A')}, Sentiment={nvda.get('Sentiment', 'N/A')}")
            else:
                print(f"  (NVDA not in current WSB trending)")
            return True
        elif resp.status_code == 401:
            print(f"✗ UNAUTHORIZED - Check API key")
            return False
        else:
            print(f"✗ FAILED - Status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("API READINESS VERIFICATION")
    print("=" * 60)

    results = {}

    # REST API tests
    results["Alpaca REST"] = test_alpaca_rest()
    results["Alpaca Options"] = test_alpaca_options_chain()
    results["ORATS"] = test_orats()
    results["Finnhub"] = test_finnhub()
    results["Quiver"] = test_quiver()

    # WebSocket test
    try:
        results["Alpaca WS"] = asyncio.run(test_alpaca_websocket())
    except Exception as e:
        print(f"✗ WebSocket test error: {e}")
        results["Alpaca WS"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("✓ ALL TESTS PASSED - Ready for paper trading!" if all_passed else "✗ SOME TESTS FAILED - Review errors above"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
