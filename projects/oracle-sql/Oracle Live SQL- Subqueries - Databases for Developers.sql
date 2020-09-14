--0. Data Setup
create table bricks (
  brick_id integer,
  colour   varchar2(10)
);

create table colours (
  colour_name           varchar2(10),
  minimum_bricks_needed integer
);

insert into colours values ( 'blue', 2 );
insert into colours values ( 'green', 3 );
insert into colours values ( 'red', 2 );
insert into colours values ( 'orange', 1);
insert into colours values ( 'yellow', 1 );
insert into colours values ( 'purple', 1 );

insert into bricks values ( 1, 'blue' );
insert into bricks values ( 2, 'blue' );
insert into bricks values ( 3, 'blue' );
insert into bricks values ( 4, 'green' );
insert into bricks values ( 5, 'green' );
insert into bricks values ( 6, 'red' );
insert into bricks values ( 7, 'red' );
insert into bricks values ( 8, 'red' );
insert into bricks values ( 9, null );

commit;

--1. Introduction
select * from bricks;

select * from colours;

--2. Inline Views
select * from (
  select * from bricks
)

select * from (
  select colour, count(*) c
  from   bricks
  group  by colour
) brick_counts

/*
-which colours meet the minimum?
*/

select * from (
  select colour, count(*) c
  from   bricks
  group  by colour
) brick_counts
join   colours
on     brick_counts.colour = colours.colour_name
and    brick_counts.c < colours.minimum_bricks_needed

--3. Try it!
select * from (
    select colour, min(brick_id) as min_brick_id, max(brick_id) as max_brick_id
    from bricks
    group by colour
) brick_colours

--4. Nested Subqueries
select * from colours c
where  c.colour_name in (
  select b.colour from bricks b
);

/*
-select the bricks with colours
*/

select * from colours c
where  exists (
  select null from bricks b
  where  b.colour = c.colour_name
);

select * from colours c
where  c.colour_name in (
  select b.colour from bricks b
  where  b.brick_id < 5
);

/*
-select bricks with colours and id less than 5
*/

select * from colours c
where  exists (
  select null from bricks b
  where  b.colour = c.colour_name 
  and    b.brick_id < 5
);

--5. Correlated vs. Uncorrelated
select * from colours
where  exists (
  select null from bricks
);

/*
-correlated subquery = joins to table from parent query
-uncorrelated = does not join to table from parent query
-exists returns only rows from parent query
-different than in
-exists subquery can select anything as it is irrelevant
-to find all colours with at least 1 brick of the same colour, join in subquery needed
*/

select * from colours
where  exists (
  select 1 from bricks
);

--6. NOT IN vs NOT EXISTS
select * from colours c
where  not exists (
  select null from bricks b
  where  b.colour = c.colour_name
);

/*
-find all colours without a brick
*/

select * from colours c
where  c.colour_name not in (
  select b.colour from bricks b
);

/*
-no data returned as there is a brick with a null colour
*/

select * from colours c
where c.colour_name not in (
  'red', 'green', 'blue', 
  'orange', 'yellow', 'purple',
  null
);

/*
-true NOT IN requires ALL parent table rows to return false
-recall null cannot return true/false
-null returns unknown
-use NOT EXISTS or where to ignore null in subquery
*/

select * from colours c
where c.colour_name not in (
  select b.colour from bricks b
  where  b.colour is not null
);

--7.Try it!
select * from bricks b
where  b.colour in (
   select colour_name
   from colours c
   where c.minimum_bricks_needed = 2
);

--8. Scalar Subqueries
select colour_name, (
         select count(*) 
         from   bricks b
         where  b.colour = c.colour_name
         group  by b.colour
       ) brick_counts
from   colours c;

/*
-scalar subqueries return only 1 col and max 1 row
-count # of bricks by colour
-nulls are returned
*/

select colour_name, nvl ( (
         select count(*) 
         from   bricks b
         where  b.colour = c.colour_name
         group  by b.colour
       ), 0 ) brick_counts
from   colours c;

/*
-to show zero instead of null, use NVL or coalesce
*/

select colour_name, coalesce ( (
         select count(*) 
         from   bricks b
         where  b.colour = c.colour_name
         group  by b.colour
       ), 0 ) brick_counts
from   colours c;

select c.colour_name, (
         select count(*) 
         from   bricks b
         group  by colour
       ) brick_counts
from   colours c;

/*
-this query returns 4 counts which will not work for a scalar
-need to join bricks with colours in subquery (i.e. correlate)
-HAVING can use scalar instead of join
*/

select colour, count(*) count
from   bricks b
group  by colour
having count(*) < (
  select c.minimum_bricks_needed 
  from   colours c
  where  c.colour_name = b.colour
);

--9. Try it!
select c.colour_name, (
    select min(brick_id)
    from   bricks b
    where  b.colour = c.colour_name
    group by colour
    ) min_brick_id
from   colours c
where  c.colour_name is not null;

--10. Common Table Expressions
with brick_colour_counts as (
  select colour, count(*) 
  from   bricks
  group  by colour
) 
  select * from brick_colour_counts ;

--11. CTEs: Reusable Subqueries
select c.colour_name, 
       c.minimum_bricks_needed, (
         select avg ( count(*) )
         from   bricks b
         group  by b.colour
       ) mean_bricks_per_colour
from   colours c
where  c.minimum_bricks_needed < (
  select count(*) c
  from   bricks b
  where  b.colour = c.colour_name
  group  by b.colour
);

/*
-group by colour appears twice
-assign name with CTE e.g. brick_counts bc
*/

with brick_counts as (
  select b.colour, count(*) c
  from   bricks b
  group  by b.colour
)
  select c.colour_name, 
         c.minimum_bricks_needed, (
           select avg ( bc.c )
           from   brick_counts bc
         ) mean_bricks_per_colour
  from   colours c
  where  c.minimum_bricks_needed < (
    select bc.c
    from   brick_counts bc
    where  bc.colour = c.colour_name
  );
  
--12. Literate SQL
select brick_id 
from   bricks 
where  colour in ('red', 'blue');

select colour
from   bricks
group  by colour  
having count (*) < (
  select avg ( colour_count ) 
  from   (
    select colour, count (*) colour_count
    from   bricks
    group  by colour  
  )
);

/*
Count the bricks by colour
Take the average of these counts
Return those rows where the value in step 1 is greater than in step 2

-step 1 is at the bottom
-solution is to use CTE
*/

with brick_counts as ( 
  -- 1. Count the bricks by colour
  select b.colour, count(*) c
  from   bricks b
  group  by b.colour
), average_bricks_per_colour as ( 
  -- 2. Take the average of these counts
  select avg ( c ) average_count
  from   brick_counts
)
  select * from brick_counts bc  
  join   average_bricks_per_colour ac
  -- 3. Return those rows where the value in step 1 is less than in step 2
  on     bc.c < average_count; 
  
--13. Testing Subqueries
with brick_counts as (
  select b.colour, count(*) count
  from   bricks b
  group  by b.colour
), average_bricks_per_colour as (
  select avg ( count ) average_count
  from   brick_counts
)
  select * from brick_counts bc;
  
--much more robust than inline views

--14. Try it!
--count how many rows there are in colours
with colour_count as (
    select count(*) as count
    from colours c
)
  select * from colour_count;