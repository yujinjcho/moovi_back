CREATE TABLE movies (
    rotten_id   text primary key,
    url_handle  text,
    description text,
    year        integer,
    title       text,
    image_url   text
);

CREATE TABLE ratings (
    rating_id   serial primary key,
		user_id     text not null,
		rotten_id   text references movies(rotten_id),
		rating	    text not null 	
);
