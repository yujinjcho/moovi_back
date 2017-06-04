CREATE TABLE movies (
    movie_id    serial primary key,
    url_handle  text,
    description text,
    year        integer,
    title       text,
    image_url   text
);

CREATE TABLE ratings (
    rating_id   serial primary key,
		user_id     text not null,
		movie_id    serial references movies(movie_id),
		rating	    text not null 	
);
