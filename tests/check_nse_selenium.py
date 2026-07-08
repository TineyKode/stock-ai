"""
Test NSE scraping using Selenium (for dynamic content).
Requires: pip install selenium
Also requires ChromeDriver to be installed and in PATH.
"""
try:
    from selenium import webdriver
except ImportError:
    print("selenium not installed. Run: pip install selenium")
    exit(1)

try:
    driver = webdriver.Chrome()
    driver.get("https://live.mystocks.co.ke")
    print(driver.page_source[:500])  # Inspect dynamic content
finally:
    driver.quit()
