select count(*) from diet_programs dp inner join diet_program_meals dpm 
on dp.id = dpm.diet_program_id;

select * from diet_programs where id = 5637179093;
select * from diet_program_meals dpm where dpm.diet_program_id  = 5637179093;

select count(*) from diet_program_meals dpm;

select * from diet_program_meal_items dpmi;

select * from diet_program_items dpi;

select * from diet_program_items dpi 
left join diet_program_meal_items dpmi
on dpi.id = dpmi.diet_program_item_id
where dpmi.diet_program_item_id is null;

select ca.id, ca.ax_id, ca.block, ca.street, ca.building, 
ca.flat, ca.floor, ca.is_default, ca.type, g.name_en as governorate_name,
c.name_en as city_name, c.from_time as city_fromt_tim, c.to_time as city_to_time,
n.name_en as nationality_name
from customers_address ca 
inner join governorates g on ca.governorate_id = g.id
inner join cities c on c.id = ca.city_id
inner join customers c2 on c2.id = ca.customer_id
inner join nationalities n on c2.nationality_id = n.id
where ca.deleted_at is null and c2.deleted_at is null;


select c.username, c.phone, c.email, c.gender, c.date_of_birth, c.height, c.weight, n.name_en as nationality from customers c left join customers_programs cp 
on c.id = cp.customer_id
inner join nationalities n on n.id = c.nationality_id
where cp.customer_id is null and c.deleted_at  is null;

select cp.*, p.* 
from customers c 
inner join customers_programs cp on c.id = cp.customer_id
inner join diet_programs dp on dp.id = cp.program_id
left join promocodes p on cp.promocode_id = p.id
where c.deleted_at is null and p.id is not null;

select * from promocodes p;

select count(*) from customers;

select * from diet_programs dp 
inner join customers_programs cp 
on dp.id = cp.program_id
where dp.master_plan_id is null;

select * from promocodes p inner join customers_programs cp 
on p.id = cp.promocode_id;

select * from promocodes p ;

SELECT p.ax_id, p.code, p.percentage, p.amount, cp.created_at, cp.status as program_status,
dp.name_en as diet_program, dp.total_amount as diet_program_amount
  FROM promocodes p 
  INNER JOIN customers_programs cp ON p.ax_id = cp.promocode_id 
  INNER JOIN diet_programs dp on dp.id = cp.program_id
  WHERE cp.paid = 1 AND cp.paid_amount <= 0 AND cp.customer_id <> 4;


select cp.*, dp.name_en as diet_program_name, mp.name_en as master_plan_name  from customers c 
                                               inner join customers_programs cp on c.id = cp.customer_id
                                               inner join diet_programs dp on dp.id = cp.program_id
                                               left join masterplans mp on mp.id = dp.master_plan_id
                                               where c.deleted_at is null and cp.customer_id <> 4;

select cpc.customer_id, cpc.customer_program_id, count(*) total from customers_program_calories cpc
inner join diet_program_calories dpc on cpc.diet_program_calorie_id  = dpc.id
group by cpc.customer_id, cpc.customer_program_id having total > 1;

select *
from customers_program_calories cpc
inner join diet_program_calories dpc on cpc.diet_program_calorie_id  = dpc.id
where cpc.customer_program_id  = 5637274377;

select
	cpm.customer_program_id,
	cpm.customer_id,
	cpm.program_day,
	cpm.calender_date,
	dpi.id as item_id,
	dpi.name_en as item_name,
	dpm.meal_code as meal_code,
	dpm.name_en as meal_name,
	cpm.rate
from
	customers_programs cp
inner join customers c on
	cp.customer_id = c.id
inner join customers_program_menus cpm on
	cp.id = cpm.customer_program_id
inner join diet_program_items dpi on
	cpm.program_item_id = dpi.id
inner join diet_program_meals dpm on
	cpm.program_meal_id = dpm.id
where
	c.deleted_at is null
	and c.id <> 4;

select
	dp.id as program_id,
	dp.name_en as program,
	dpmi.program_day,
	dpm.meal_code,
	dpm.name_en as meal_name,
	dpi.id as item_id,
	dpi.name_en as item_name_en,
	dpi.name as item_name_ar,
	dpi.created_at as item_created_at
from
	diet_program_items dpi
left join
customers_program_menus cpm on
	dpi.id = cpm.program_item_id
inner join diet_program_meal_items dpmi 
on
	dpi.id = dpmi.diet_program_item_id
inner join diet_program_meals dpm on
	dpm.id = dpmi.diet_program_meal_id
inner join diet_programs dp on
	dpm.diet_program_id = dp.id
where
	cpm.id is null;
