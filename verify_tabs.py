import asyncio
from playwright.async_api import async_playwright
import time

async def save_screenshot(page, filename):
    await page.screenshot(path=filename)
    print(f"Saved screenshot to {filename}")

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to http://localhost:8050...")
        await page.goto("http://localhost:8050")
        
        print("Waiting for tabs to load...")
        await page.wait_for_selector("#report-tabs", timeout=10000)
        
        print("Clicking Auto ARIMA tab...")
        await page.locator('.tab:has-text("Auto ARIMA")').click()
        await page.wait_for_timeout(1000)
        await save_screenshot(page, "auto_arima_initial.png")

        print("Clicking 'Run Auto ARIMA'...")
        await page.locator("#btn-auto-arima").click()
        
        print("Sleeping 500ms to capture 'Running' text...")
        await page.wait_for_timeout(500)
        await save_screenshot(page, "auto_arima_loading.png")
        
        print("Waiting 5 seconds for computation to finish...")
        await page.wait_for_timeout(5000)
        await save_screenshot(page, "auto_arima_done.png")
        
        print("Clicking 'Data Overview' tab...")
        await page.locator('.tab:has-text("Data Overview")').click()
        await page.wait_for_timeout(1000)
        await save_screenshot(page, "data_overview.png")

        print("Clicking BACK to 'Auto ARIMA' tab...")
        await page.locator('.tab:has-text("Auto ARIMA")').click()
        await page.wait_for_timeout(1000)
        await save_screenshot(page, "auto_arima_returned.png")

        await browser.close()

asyncio.run(run())
