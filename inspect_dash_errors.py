import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Capture console messages
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))
        # Capture unhandled exceptions
        page.on("pageerror", lambda exc: print(f"PAGE ERROR: {exc}"))

        print("Navigating to http://localhost:8050...")
        await page.goto("http://localhost:8050")
        
        # Wait for the app to load
        await page.wait_for_selector("#report-tabs", timeout=10000)
        
        print("Switching to Auto ARIMA tab...")
        # Since it's a Dash tab, we click the tab with text "Auto ARIMA"
        await page.locator('.tab:has-text("Auto ARIMA")').click()
        await page.wait_for_timeout(1000)

        print("Clicking Run Auto ARIMA button...")
        await page.locator("#btn-auto-arima").click()
        
        print("Waiting 3 seconds for callback to start...")
        await page.wait_for_timeout(3000)
        
        print("Switching to Data Overview tab...")
        await page.locator('.tab:has-text("Data Overview")').click()
        await page.wait_for_timeout(1000)

        print("Switching BACK to Auto ARIMA tab...")
        await page.locator('.tab:has-text("Auto ARIMA")').click()
        await page.wait_for_timeout(2000)
        
        print("Checking if plot exists...")
        plot_count = await page.locator("#auto-arima-output .js-plotly-plot").count()
        print(f"Plotly plots found: {plot_count}")

        await browser.close()

asyncio.run(run())
