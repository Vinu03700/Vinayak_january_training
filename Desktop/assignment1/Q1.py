# Q1: Message Counter (Immutable Parameter)
# This function demonstrates the behavior of immutable parameters with default values.
# The count parameter is an integer (immutable), and the default value is 0.
# Each function call starts with count=0, increments it, prints, and returns.
# Across multiple calls, the count does not persist because integers are immutable,
# and the default parameter is evaluated only once, but since it's immutable, it behaves as expected (resets).

def count_message(msg, count=0):
    """
    Accepts a message and increments the count.
    Prints the message and updated count, then returns the count.
    """
    count += 1  # Increment the count
    print(f"Message: {msg}, Updated Count: {count}")
    return count

# Sample calls to observe behavior
count_message("heya")
count_message("hello")
count_message("hi")
