from pprint import pprint
import psycopg2

import config

db = None

def db_connect():
    global db
    if not db:
        db = psycopg2.connect(**config.db_config)
    return db


def retrieve_svm(user):
    conn = db_connect()
    query = """

        select 
          r.rating, 
          q.svm_text, 
          q.rotten_id 
        from (
          select 
            rotten_id, 
            array_to_string(array_agg(user_num || ':' || rating order by user_num), ' ') as svm_text 
          from ratings_test 
          where user_num != %s
          group by rotten_id
        ) as q 
        left join ratings_test r on r.rotten_id = q.rotten_id and r.user_num = %s;

    """
    with conn.cursor() as cur:
        cur.execute(query, (user, user))
        results = cur.fetchall()

    return results

def box_office():

    conn = db_connect()
    query = """
        SELECT 
          UNNEST(movies)
        FROM (
          SELECT movies
          FROM box_office
          ORDER BY 
             date_created DESC
          LIMIT 1
        ) i
    """
    with conn.cursor() as cur:
        cur.execute(query)
        results = cur.fetchall()
    return set(x[0] for x in results)
    

def create_train_svm(user_num):
    query = """
        EXPLAIN ANALYZE
        select 
            um.rating, 
            um.rotten_id, 
            array_to_string(array_agg(r.user_num || ':' || r.rating order by user_num), ' ') as svm_text 
        from (
            select 
                rotten_id, 
                rating 
            from ratings_test 
            where user_num = %s
        ) as um 
        left join ratings_test r on r.rotten_id = um.rotten_id and r.user_num != %s
        group by um.rotten_id, um.rating;

    """
    conn = db_connect()
    with conn.cursor() as cur:
        cur.execute(query, (user_num, user_num))
        result = cur.fetchall()

    return result


# TRAINING_SVM
# select um.rating, um.rotten_id, array_to_string(array_agg(r.user_num || ':' || r.rating order by user_num), ' ') as svm_text from (select rotten_id, rating from ratings_test where user_num = 5000) as um left join ratings_test r on r.rotten_id = um.rotten_id and r.user_num != 3 group by um.rotten_id, um.rating;

# PREDICT_SVM
# select b.movie_id, m.title, array_to_string(array_agg(r.user_num || ':' || r.rating order by user_num), ' ') as svm_text from (select unnest(movies) as movie_id from (select movies from box_office order by date_created desc limit 1) as b) as b left join ratings_test r on r.rotten_id = b.movie_id and r.user_num != 2000 left join movies m on m.rotten_id = b.movie_id group by b.movie_id, m.title;

if __name__ == '__main__':
    test_user = 1791
    results = create_train_svm(test_user)
    pprint(results)
