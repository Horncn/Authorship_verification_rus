from lxml import html
import requests
import re
import os

def parce():
    print('getting links')
    file = open('links_1.txt', 'w')
    pages = 1208
    url = 'https://ficbook.net/fanfiction/books/naruto?premium=0&filterDirection=&rating=&size='
    url += str(2) + '&p='
    for p in range(1, pages + 1):
        if not p % 10:
            print(p)
        cur_url = url + str(p)
        ref = requests.get(cur_url).text
        links = re.findall(r'/readfic/[0-9]*', ref)
        for link in links:
            link = link[8:]
            file.write(link + ',')
    file.close()

def count_links():
    file = open('links_1.txt', 'r')
    text = file.read()
    file.close()
    print(text.count('/'))

def download_text(id):
    url = 'https://ficbook.net/readfic' + id
    ref = requests.get(url).text
    try:
        author = re.findall(r'authors/[0-9]*', ref)[0]
    except IndexError:
        return 0
    if author:
        author = author[8:]
    tree = html.fromstring(ref)
    text = tree.xpath('//div[@id="content"]/text()')
    text = '/n'.join(text)
    if not text:
        return 0
    file = open('texts' + id + '.txt', 'w', encoding="utf-8")
    file.write(text)
    file.close()
    return author


def download_all():
    print('start downloading')
    file = open('links_1.txt')
    links = file.read().split(',')
    autors = {}
    prof_autors = {}
    counter = 0
    for link in links:
        counter += 1
        if not counter % 100:
            print('proceed: ' + str(counter) + ' links out of ' + str(len(links)))
        autor = download_text(link)
        if not autor:
            continue
        if autor not in autors:
            autors[autor] = [link]
            prof_autors[autor] = 1
        else:
            autors[autor].append(link)
            prof_autors[autor] += 1
    top_autors = sorted(prof_autors, key=lambda x: int(prof_autors[x]), reverse=True)

    db_file = open('db.csv', 'w')
    for key in top_autors:
        db_file.write(key + ',')
        for text in autors[key]:
            db_file.write('|' + text)
        db_file.write('\n')
    db_file.close()

    prof_file = open('prof.csv', 'w')
    for key in top_autors:
        prof_file.write(key + ',' + str(prof_autors[key]) + '\n')
    prof_file.close()

def delete_unique_authors():
    file = open('db.csv', 'r')
    db = file.read().split('\n')
    file.close()
    for row in db:
        if row.count('|') > 1:
            continue
        try:
            row = row.split(',')[1]
            row = row.split('|')[1][1:]
        except IndexError:
            continue
        try:
            os.remove (os.path.join(os.path.abspath(os.path.dirname(__file__)), ('texts\\' + row + '.txt')))
        except FileNotFoundError:
            print(row)
    print('succeed')

def size_meter():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'texts\\')
    links = os.listdir(path)
    file = open('text_authors.csv', 'w')
    counter = 0
    for link in links:
        counter += 1
        if not counter % 10:
            print('Done: ' + str(counter) + ' out of ' + str(len(links)))
        link = link[:-4]
        url = 'https://ficbook.net/readfic/' + link
        ref = requests.get(url).text
        author = re.findall(r'authors/[0-9]*', ref)[0]
        file.write(link + ',' + author[8:] + '\n')
    file.close()

def fix():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'texts\\')
    links = os.listdir(path)
    pre_file = open('text_authors.csv', 'r')
    db = pre_file.read()
    db = db.split('\n')
    database = {}
    counter = {}
    for d in db:
        d = d.split(',')
        try:
            database[d[0]] = d[1]
        except IndexError:
            print(d)
    file = open('text_authors_fix.csv', 'w')
    file.write('text,author\n')
    for key in database.keys():
        if database[key] in counter:
            counter[database[key]] += 1
        else:
            counter[database[key]] = 1
        if key + '.txt' in links:
            file.write(str(key) + ',' + str(database[key]) + '\n')
    file.close()
    for key in counter.keys():
        if counter[key] == 1:
            print(key)


def run():
    parce()
    count_links()
    download_all()
    delete_unique_authors()

run()
