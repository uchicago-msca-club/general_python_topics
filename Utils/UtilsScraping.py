from urllib.request import urlopen as uReq, urlretrieve
from bs4 import BeautifulSoup as soup

def get_page_soup(url):
    # Opening a connection, grabbing the page
    client = uReq(url)

    # Save the download into an object
    page_html = client.read()

    # Close the client
    client.close()

    # Clean up the HTML code
    page_soup = soup(page_html, "html.parser")

    return page_soup