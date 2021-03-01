

def prime_checker(number):

    # prime number is always greater than 1
    if number > 1:
        for i in range(2, number):
            if (number % i) == 0:

                return False
        else:
            return True

    # if the entered number is less than or equal to 1
    # then it is not prime number
    else:
        return False


for i in range(50, 71):
    if  prime_checker(i):
        print(i)
    else:
        i =+ 1
