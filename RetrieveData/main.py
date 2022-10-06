import re

#making changes

# assuming we have a Username setup previously
Username = "ckobie"

def userInput():
    # query for ambient temperature and humidity or wakeup time
    timeOrTemp = input("Do you want to know 'wakeupTime' or 'weather data'? ")
    if timeOrTemp == "wakeupTime":
        requiresUserName()
    if timeOrTemp == "weather data":
        dataTemp()
    else:
        print("invalid input")
        exit()
    return


def requiresUserName():
    # login stuff
    i = 3
    while i > 0:
        username = input("Enter username: ")
        if username == Username:
            dataTime()
        else:
            print("username is invalid or unregistered")
            i = i - 1
            print("you have ", i, " attempt(s) left")
    exit()
    return


def dataTime():
    formatting = input("do you want the time by 'date' or 'days/weeks' ago? ")
    if formatting == "date":
        pattern = "^[0-9]{2}/[0-9]{2}/[0-9]{4}$"
        date = input("what date? [MM/DD/YYYY] ")
        if re.match(pattern, date):
            print("getting data")
            #insert getting data code here
        else:
            print("invalid format")
        exit()
    if formatting == "days/weeks":
        dw = input("do you want 'days' or 'weeks'? ")
        n = input("how many " + dw + " ago? ")
        print("retrieving data for " + n + " " + dw + " ago")
        #insert getting data code here
    else:
        print("invalid input")
    return


def dataTemp():
    print("getting data")
    return


def returnDataAsJSON():
    return


userInput()
