--0. Data Setup
create table bricks (
  brick_id integer,
  colour   varchar2(10),
  shape    varchar2(10),
  weight   integer
);

insert into bricks values ( 1, 'blue', 'cube', 1 );
insert into bricks values ( 2, 'blue', 'pyramid', 2 );
insert into bricks values ( 3, 'red', 'cube', 1 );
insert into bricks values ( 4, 'red', 'cube', 2 );
insert into bricks values ( 5, 'red', 'pyramid', 3 );
insert into bricks values ( 6, 'green', 'pyramid', 1 );

commit;

--1. Introduction
select count(*) from bricks;

select count(*) over () from bricks;

select b.*, 
       count(*) over () total_count 
from   bricks b;

--2. Partition By
select colour, count(*), sum ( weight )
from   bricks
group  by colour;

select b.*, 
       count(*) over (
         partition by colour
       ) bricks_per_colour, 
       sum ( weight ) over (
         partition by colour
       ) weight_per_colour
from   bricks b;

/*
Group by produces aggregate vs partition produces statistic per row 
e.g. 1 row with the count vs all rows with the count
*/


--3. Try it!
select b.*, 
       count(*) over (
         partition by shape
       ) bricks_per_shape, 
       median ( weight ) over (
         partition by shape
       ) median_weight_per_shape
from   bricks b
order  by shape, weight, brick_id;

--4. Order By
select b.*, 
       count(*) over (
         order by brick_id
       ) running_total, 
       sum ( weight ) over (
         order by brick_id
       ) running_weight
from   bricks b;

--5. Try it!
select b.brick_id, b.weight,
       round ( avg ( weight ) over (
         order by brick_id


       ), 2 ) running_average_weight
from   bricks b
order  by brick_id;

--6. Partition By + Order By
select b.*, 
       count(*) over (
         partition by colour
         order by brick_id
       ) running_total, 
       sum ( weight ) over (
         partition by colour
         order by brick_id
       ) running_weight
from   bricks b;

--7. Windowing Clause
select b.*, 
       count(*) over (
         order by weight
       ) running_total, 
       sum ( weight ) over (
         order by weight
       ) running_weight
from   bricks b
order  by weight;

/*
By default the order by returns all the rows with a value less than or equal to that of the current row.
This includes values from rows after the current row.
This is not expected behaviour as running totals should not be summing future values.
*/

select b.*, 
       count(*) over (
         order by weight
         rows between unbounded preceding and current row
       ) running_total, 
       sum ( weight ) over (
         order by weight
         rows between unbounded preceding and current row
       ) running_weight
from   bricks b
order  by weight;

/*
Rows can have the same weight can result in different running totals.
Thus there has to be another unique column value to sort by to reproduce the same order.
*/

select b.*, 
       count(*) over (
         order by weight, brick_id
         rows between unbounded preceding and current row
       ) running_total, 
       sum ( weight ) over (
         order by weight, brick_id
         rows between unbounded preceding and current row
       ) running_weight
from   bricks b
order  by weight, brick_id;

--8. Sliding Windows
select b.*, 
       sum ( weight ) over (
         order by weight
         rows between 1 preceding and current row
       ) running_row_weight, 
       sum ( weight ) over (
         order by weight
         range between 1 preceding and current row
       ) running_value_weight
from   bricks b
order  by weight, brick_id;

/*
aka "moving weight" sum
*/

select b.*, 
       sum ( weight ) over (
         order by weight
         rows between 1 preceding and 1 following
       ) sliding_row_window, 
       sum ( weight ) over (
         order by weight
         range between 1 preceding and 1 following
       ) sliding_value_window
from   bricks b
order  by weight;

select b.*, 
       count (*) over (
         order by weight
         range between 2 preceding and 1 preceding 
       ) count_weight_2_lower_than_current, 
       count (*) over (
         order by weight
         range between 1 following and 2 following
       ) count_weight_2_greater_than_current
from   bricks b
order  by weight;

--9. Try It!
select b.*, 
       min ( colour ) over (
         order by brick_id
         rows between 2 preceding and 1 preceding
       ) first_colour_two_prev, 
       count (*) over (
         order by weight
         range between current row and 1 following
       ) count_values_this_and_next
from   bricks b
order  by weight;

--10. Filtering Analytic Functions
select colour from bricks
group  by colour
having count(*) >= 2;


/*
select colour from bricks
where  count(*) over ( partition by colour ) >= 2;

--note
--this produces an error as oracle db will first filter the query 
--with the where clause and then apply the partition

the solution is to use a subquery so that the partition occurs before the filter
*/

select * from (
  select b.*,
         count(*) over ( partition by colour ) colour_count
  from   bricks b
)
where  colour_count >= 2;

--11. Try It!
with totals as (
  select b.*,
         sum ( weight ) over ( 
           partition by shape
         ) weight_per_shape,
         sum ( weight ) over ( 
           order by brick_id, weight
           rows between unbounded preceding and current row
         ) running_weight_by_id
  from   bricks b
)
  select * from totals
  where weight_per_shape > 4
  and running_weight_by_id > 4
  order  by brick_id
  
--12. More Analytic Functions
select brick_id, weight, 
       row_number() over ( order by weight ) rn, 
       rank() over ( order by weight ) rk, 
       dense_rank() over ( order by weight ) dr
from   bricks;

/*
Rank - Rows with the same value in the order by have the same rank. The next row after a tie has the value N, where N is its position in the data set.
Dense_rank - Rows with the same value in the order by have the same rank, but there are no gaps in the ranks
Row_number - each row has a new value
*/

select b.*,
       lag ( shape ) over ( order by brick_id ) prev_shape,
       lead ( shape ) over ( order by brick_id ) next_shape
from   bricks b;

/*
-get prior or next value
*/

select b.*,
       first_value ( weight ) over ( 
         order by brick_id 
       ) first_weight_by_id,
       last_value ( weight ) over ( 
         order by brick_id 
       ) last_weight_by_id
from   bricks b;

/*
-get first/last value
-recall default windowing clause stops at the current row. 
-need explicit "unbounded following"
*/

select b.*,
       first_value ( weight ) over ( 
         order by brick_id 
       ) first_weight_by_id,
       last_value ( weight ) over ( 
         order by brick_id 
         range between current row and unbounded following
       ) last_weight_by_id
from   bricks b;