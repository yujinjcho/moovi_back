import sys
import json
from optparse import OptionParser
import psycopg2

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", metavar="FILE",
                  help="name of upload file")
(options, args) = parser.parse_args()

conn = psycopg2.connect(
    dbname='d4re7d0e0r1t8v',
    user='iqtoybmgmvwkts',
    password='a26080926d1e5e819ab81ed346320d1e1acc32bd040053ebde3f6b8192cf7754',
    port='5432',
    host='ec2-107-20-226-93.compute-1.amazonaws.com')
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
