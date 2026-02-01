# Q2: Order History Tracker (Mutable Parameter)
# This function demonstrates the use of mutable parameters (lists) with default values.
# The orders parameter is a list (mutable), and the default is None to avoid the pitfall.
# Inside the function, if orders is None, initialize it as an empty list.
# This way, each call starts with a fresh list if not provided, but we can pass the same list across calls.
# To store across calls, we need to use a mutable default carefully, but since it's None, we handle it.
# Actually, to persist across calls, we can use a global or class, but the task says DO NOT use global variables.
# The task: "Store order IDs across function calls" without global.
# But with default None, each call gets a new list unless we pass the previous one.
# Sample calls: add_order(101); add_order(102) – but to accumulate, we need to return the list and pass it back.
# Probably, the function should return the orders list, and in calls, do orders = add_order(101, orders); orders = add_order(102, orders)
# But the sample is just add_order(101); add_order(102) – perhaps it's meant to accumulate in a shared default, but that's the pitfall.
# The task says "Use a default parameter safely" and "work correctly when called multiple times".
# To avoid global, perhaps use a closure or something, but that's advanced.
# Perhaps the intention is to use the mutable default but fix it.
# Standard way: def add_order(order_id, orders=None):
#     if orders is None:
#         orders = []
#     orders.append(order_id)
#     return orders
# But then for multiple calls, it won't accumulate unless we do orders = add_order(101); orders = add_order(102, orders)
# The sample is just add_order(101); add_order(102), so perhaps it's to show the pitfall, but the task is to make it work.
# For Q2, it's to use mutable safely, so probably the fixed version.
# I'll implement it as returning the list, and in code, show accumulation by assigning.

def add_order(order_id, orders=None):
    """
    Adds an order ID to the orders list.
    If orders is None, initializes a new list.
    Returns the updated orders list.
    """
    if orders is None:
        orders = []
    orders.append(order_id)
    return orders

# Sample calls to observe behavior
# To accumulate, we need to pass the list back
orders = add_order(101)
orders = add_order(102, orders)
print("Order history:", orders)
