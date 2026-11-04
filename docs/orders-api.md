# Orders API Reference

## Create an order
Send a POST request to /orders with a list of items and a payment method. A successful request returns status 201 and the new order id.

## Get an order
Send a GET request to /orders/{id}. It returns status 200 with the order details. If the id does not exist, it returns status 404.

## Cancel an order
Send a DELETE request to /orders/{id}. An order can only be cancelled before it is shipped. After that, the API returns status 409.

## Authentication
Every request needs an API key in the header named x-api-key. A missing or wrong key returns status 401.

## Rate limits
Each API key can send 100 requests per minute. Above that, the API returns status 429 and the client must wait before trying again.