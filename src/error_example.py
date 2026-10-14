# Try to execute code that might cause an error
try:

    # This will cause an error
    number = 10 / 0

# Handle the error
except ZeroDivisionError:

    print("You cannot divide a number by zero")