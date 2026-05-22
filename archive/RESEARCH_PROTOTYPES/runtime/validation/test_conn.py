import asyncio
import aiohttp
import time

async def test():
    print("Client started")
    async with aiohttp.ClientSession() as session:
        print("Session opened")
        try:
            async with session.get("http://localhost:8000/v1/models") as resp:
                print(f"Status: {resp.status}")
                print(await resp.text())
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
