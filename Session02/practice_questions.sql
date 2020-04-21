# PRACTICE QUESTIONS FOR NORTHWIND DATABASE

use northwind;
select * from order_details;

# QUESTION 1
# We have a table called Shippers. Return all the fields for all the shippers
select * from shippers;

# QUESTION 2
# Suppose we want to show all the orders from Brazil, Mexico, Argentina & Venezuela
SELECT id, customer_id, ship_country_region
FROM orders
WHERE ship_country_region IN ('Brazil', 'Mexico', 'Argentina', 'Venezuela');

# QUESTION 3
# Show a list of all the different values in the Customers table for ContactTitles. 
# Also include a count for each ContactTitle.
SELECT job_title, COUNT(distinct id) AS n_customers
FROM customers
GROUP BY job_title
ORDER BY total_customers DESC;

# QUESTION 4
# Distribution frequency of customer's states
SELECT t.n_customers, count(*) as n_states
FROM
	(SELECT state_province, count(*) as n_customers
	 FROM customers
     GROUP BY state_province) as t
GROUP BY t.n_customers
ORDER BY n_states DESC;

# QUESTION 5
# Find all the orders including customer info, order details and status.
# Keep only orders whose full information is available
SELECT	
    orders.id as order_id, order_det.status_name as order_status, orders.order_date,
    order_det.product_id, order_det.product_name, order_det.category,
    customers.company, customers.state_province,
    sum(order_det.quantity * order_det.unit_price - order_det.discount) as total_amount
FROM 
	(orders
		INNER JOIN customers ON orders.customer_id = customers.id
		INNER JOIN 
			(SELECT order_details.order_id, order_details.product_id, order_details.quantity, 
					order_details.unit_price, order_details.discount, order_details.status_id, 
					order_details_status.status_name, products.product_name, products.category
             FROM order_details 
				  LEFT JOIN order_details_status ON order_details.status_id = order_details_status.id
                  LEFT JOIN products ON order_details.product_id = products.id) as order_det
                  ON order_det.order_id = orders.id);
                  
      
# QUESTION 6
# Find the number of orders not shipped for each salesperson and compared against the total number of orders per salesperson
# Keep only those salesperson for whom > 50% of all orders werxwe not shipped

SELECT sale.id, sale.last_name, sale.first_name, sale.job_title, count(*) as n_orders, 
	 sum(sale.not_shipped) as n_not_shipped, round(sum(sale.not_shipped) / count(*) * 100) as pct_not_shipped
FROM (SELECT e.id, e.last_name, e.first_name, e.job_title, o.order_date, o.shipped_date,
			 CASE WHEN o.shipped_date IS NULL and o.order_date is NOT NULL THEN 1 ELSE 0 END as not_shipped
	  FROM employees as e
	  LEFT JOIN orders as o ON o.employee_id = e.id) as sale
GROUP BY sale.id
HAVING pct_not_shipped > 50
ORDER BY pct_not_shipped DESC


