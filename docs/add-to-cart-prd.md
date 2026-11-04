# Add to Cart Feature: Product Requirements

## Purpose
Customers must be able to collect items in a cart before they pay.

## Quantity limits
A customer can add at most 10 units of the same item. A cart can hold at most 50 different items.

## Out of stock items
An item that is out of stock cannot be added to the cart. The Add to Cart button is replaced by a message saying the item is unavailable.

## Prices
The price shown in the cart includes tax. The price is checked again at checkout, and the customer is told if it has changed.

## Saved carts
A cart of a signed-in customer is saved for 7 days. A cart of a visitor who is not signed in is lost when the browser is closed.

## Duplication Of Items
If any item is already added into the cart and user is adding it again then it should update the quantity. It should not show same item two times. The limit of 10 units per item still applies when an item is added again. If the new total would be more than 10, the quantity is set to 10 and the customer sees a message that explains the limit.