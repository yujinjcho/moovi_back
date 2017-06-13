insert into movies ( rotten_id, url_handle, description, year, title, image_url ) 
select values::json->>'rt_id' as rotten_id, 
       values::json->>'movie_url_handle' as url_handle,
       values::json->>'description' as description,
       (values::json->>'year')::integer as year,
       values::json->>'movie_title' as title,
       values::json->>'poster_url' as image_url
       from temp_movies


