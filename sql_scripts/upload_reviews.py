import sys
import json
import psycopg2
from optparse import OptionParser

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
    
            # upload_query = """INSERT INTO ratings (user_id, movie_id, rating) 
            #                   VALUES (%s, %s, %s)
            # """
            # upload_data = (user_id, movie_id, rating)
            # cur.execute(upload_query, upload_data)
       
        # count += 1 
        # print "Uploaded {} of {}".format(count, total)
   
conn.commit()
cur.close()
conn.close()
print "finished uploading"
