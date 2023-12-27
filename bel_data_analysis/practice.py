# 1) No is a Prime or not
'''num = int(input("Enter the num: "))
flag = False

if num == 1:
    print("num is not a prime no")
else:
    for i in range(2, num):
        if num % i == 0:
            flag = True
            break
if flag:
    print("prime no")
else:
    print("not a prime number")'''

# 2) Factorial of a number
'''num = int(input("enter the number:"))
factorial = 1
if num<0:
    print("factorial doesen't exist")
elif num==0:
    print("facterial is 1")
else:
    for i in range(1, num+1):
        factorial = factorial*i
    print("factorial of number is:", factorial)'''

# 3) Swapping a Number:
'''x = int(input("enter x: "))
y = int(input("enter y: "))
temp = x
x = y
y = temp
print(f"value of x is {x} and value of y is {y}")'''

# 4) armstrong number
'''num = int(input("eneter the no:"))
order = len(str(num))
sum = 0
temp = num
while temp > 0:
    digit = temp%10
    sum = sum + digit**order
    temp = temp//10
if sum == num:
    print("armstrong number")
else:
    print("not a Armstrong number")'''

# 5)