import sys
import json
import psycopg2
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", metavar="FILE",
                          help="name of upload file")
(options, args) = parser.parse_args()

conn = psycopg2.connect("dbname=movie_rec")
cur = conn.cursor()
count = 0

with open(options.input) as f:
    for line in f:
    
        data = json.loads(line.rstrip())
        url_handle = data['movie']
    
        movie_id_query = "SELECT movie_id from movies WHERE url_handle = %s"
        movie_id_data = (url_handle,)
        cur.execute(movie_id_query, movie_id_data)
        result = cur.fetchone()
        
        if result:
            movie_id = result[0]
            user_id = data['critic']
            rating = data['rating']
    
            upload_query = """INSERT INTO ratings (user_id, movie_id, rating) 
                              VALUES (%s, %s, %s)
            """
            upload_data = (user_id, movie_id, rating)
            cur.execute(upload_query, upload_data)
        
        count += 1
        if count % 10000 == 0:
            print count
   
conn.commit()
cur.close()
conn.close()
print "finished uploading"
