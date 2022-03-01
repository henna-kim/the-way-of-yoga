import requests
from bs4 import BeautifulSoup

response = requests.get('https://www.tripadvisor.in/Hotels-g187147-Paris_Ile_de_France-Hotels.html')
bsobj = BeautifulSoup(response.content,'lxml')


# Grab all links

links = []

for review in bsobj.findAll('a',{'class':'review_count'}):
  a = review['href']
  a = 'https://www.tripadvisor.in'+ a
  links.append(a)
print(links)
