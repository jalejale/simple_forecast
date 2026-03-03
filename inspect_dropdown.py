import asyncio
from playwright.async_api import async_playwright
import time

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8050")
        
        # Wait for the dropdown to be visible
        await page.wait_for_selector(".dash-dropdown")
        
        # Click the dropdown to open the menu
        await page.click(".dash-dropdown")
        await page.wait_for_timeout(500)  # Wait for animation
        
        # Extract the entire HTML of the opened dropdown
        dropdown_html = await page.evaluate("() => document.querySelector('.dash-dropdown').outerHTML")
        print("DROPDOWN HTML DUMP:\n")
        print(dropdown_html)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
