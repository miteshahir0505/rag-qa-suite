# Checkout Feature: Product Requirements

## Purpose
Customers must be able to pay for the items in their cart quickly and safely.

## Payment methods
The checkout accepts credit cards, debit cards and bank transfers. Cash on delivery is not supported in version 1.

## Discount codes
A customer can apply one discount code per order. A second code replaces the first one. Codes are not case sensitive.

## Order limits
The maximum order value is 5000 dollars. An order above this limit is rejected with a message asking the customer to contact support.

## Failed payments
If a payment fails, the cart is kept for 24 hours so the customer can try again. The customer must see a clear message that explains why the payment failed.

## Refunds
A customer can request a refund within 5 days of delivery. Refunds are paid back to the original payment method within 5 working days.