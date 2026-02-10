from urllib.request import urlopen
import re


#quiz 1
url = "http://olympus.realpython.org/profiles/dionysus"
html_page = urlopen(url)
html_text = html_page.read().decode("utf-8")

for string in ["Name: ", "Favorite Color:"]:
    string_start = html_text.find(string)
    text_start = string_start + len(string)

    next_html_tag = html_text[text_start:].find("<")
    text_end = text_start + next_html_tag

    raw_text = html_text[text_start : text_end]
    clean_text = raw_text.strip("\r\n\t")

    #print(clean_text)


#Quiz 2
from urllib.request import urlopen
from bs4 import BeautifulSoup

base_url = "http://olympus.realpython.org"

html_page = urlopen(base_url + "/profiles")
html_text = html_page.read().decode("utf-8")

soup = BeautifulSoup(html_text, "html.parser")

for link in soup.find_all("a"):
    link_url = base_url + link["href"]
    #print(link_url)


#Quiz 3
import mechanicalsoup

# 1
browser = mechanicalsoup.Browser()
url = "http://olympus.realpython.org/login"
login_page = browser.get(url)
login_html = login_page.soup

# 2
form = login_html.select("form")[0]
form.select("input")[0]["value"] = "zeus"
form.select("input")[1]["value"] = "ThunderDude"

# 3
profiles_page = browser.submit(form, login_page.url)
print(profiles_page.soup.title)