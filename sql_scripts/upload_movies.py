import sys
import json
from optparse import OptionParser
import psycopg2

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", metavar="FILE",
                  help="name of upload file")
(options, args) = parser.parse_args()

conn = psycopg2.connect("dbname=movie_rec")
cur = conn.cursor()
description = None
year = None
title = None
image_url = None

# for line in sys.stdin:
with open(options.input) as f:
    for line in f:

        data = json.loads(line.rstrip())
        url_handle = data['movie_url_handle']

        if 'description' in data:
            description = data['description']

        if 'year' in data:
            year = data['year']

        if 'movie_title' in data:
            title = data['movie_title']

        if 'poster_url' in data:
            image_url = data['poster_url']

        SQL = """INSERT INTO movies 
        (url_handle, description, year, title, image_url) 
        VALUES (%s, %s, %s, %s, %s);"""
        data = (url_handle, description, year, title, image_url)
        cur.execute(SQL, data)

        description = None
        year = None
        title = None
        image_url = None
   
conn.commit()
cur.close()
conn.close()
print "Completed uploading"
