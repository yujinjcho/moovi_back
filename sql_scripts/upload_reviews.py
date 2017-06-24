import sys
import json
import psycopg2
from optparse import OptionParser
from config import db_config


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", metavar="FILE",
                          help="name of upload file")
(options, args) = parser.parse_args()

conn = psycopg2.connect(**db_config)
cur = conn.cursor()
count = 0
total = 870796

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
            print json.dumps({'movie_id':movie_id, 'user_id': user_id, 'rating': rating})
    
   
conn.commit()
cur.close()
conn.close()
print "finished uploading"
