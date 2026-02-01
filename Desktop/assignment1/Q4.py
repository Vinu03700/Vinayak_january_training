# Q4.py
# Remove all even numbers from list

def remove_even(nums):
    result = []

    for n in nums:
        if n % 2 != 0:
            result.append(n)

    return result


# Input
nums = [1,2,3,4,5,6,7,8,9,10]
print(remove_even(nums))
