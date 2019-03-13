"""
If input number is odd, then print "weird"
If input number is even, then print "not weird"

If input is 2 or 5, then print "2" or "5" depending the number
"""

import sys

inputNum = input("Enter an integer: \n")


#Check if enter value is PROPER
try:
    savedNum = int(inputNum)
except ValueError:
    print ("Number entered is not proper")
    sys.exit()


if savedNum == 2:
    print ("Number is 2")
elif savedNum == 5:
    print ("Number is 5")
elif savedNum % 2 == 0:
    print ("even")
else:
    print ("odd")
