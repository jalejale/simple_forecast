import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8050")
        await page.wait_for_selector("#brand-dd")
        
        # Click the specific dropdown
        await page.click("#brand-dd")
        await page.wait_for_timeout(1000)
        
        # Get the HTML of the menu using aria-controls
        menu_html = await page.evaluate('''() => {
            const trigger = document.querySelector('#brand-dd');
            if (!trigger) return 'NO TRIGGER';
            const menuId = trigger.getAttribute('aria-controls');
            if (!menuId) return 'NO ARIA CONTROLS';
            const menu = document.getElementById(menuId);
            return menu ? menu.outerHTML : 'MENU NOT FOUND IN DOM';
        }''')
        
        print("DROPDOWN MENU HTML:\n")
        print(menu_html)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
