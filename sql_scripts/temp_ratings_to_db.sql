insert into ratings ( rotten_id, user_id, rating ) 
select values::json->>'movie_id' as rotten_id, 
       values::json->>'user_id' as user_id,
       values::json->>'rating' as rating
       from temp_ratings
 where values::json->>'movie_id' 
    in ( select rotten_id from movies where rotten_id = values::json->>'movie_id' )


