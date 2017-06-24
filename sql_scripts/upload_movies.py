import sys
import json
from optparse import OptionParser
import psycopg2
from config import db_config

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", metavar="FILE",
                  help="name of upload file")
(options, args) = parser.parse_args()

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

with open(options.input) as f:
    for line in f:

        data = json.loads(line.rstrip())
        url_handle = data.get('movie_url_handle')
        description = data.get('description')
        year = data.get('year')
        title = data.get('movie_title')
        image_url = data.get('poster_url')

        SQL = """INSERT INTO movies 
        (url_handle, description, year, title, image_url) 
        VALUES (%s, %s, %s, %s, %s);"""
        data = (url_handle, description, year, title, image_url)
        cur.execute(SQL, data)

   
conn.commit()
cur.close()
conn.close()
print "Completed uploading"
