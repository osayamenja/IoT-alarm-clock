# subscriber.py
import threading

import paho.mqtt.client as mqtt
import json
import time

isAlarm = False


# source
# https://stackoverflow.com/questions/5508509/how-do-i-check-if-a-string-is-valid-json-in-python#:~:text=Example%20Python%20script%20returns%20a%20boolean%20if%20a%20string%20is%20valid%20json%3A
def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # subscribe, which need to put into on_connect
    # if reconnect after losing the connection to the broker, it will continue to subscribe to the raspberry/topic topic
    client.subscribe("raspberry/topic")


def check_time():
    global isAlarm
    isAlarm = True
    for i in range(30):
        print(i)
        time.sleep(1)
    isAlarm = False


# the callback function, it will be triggered when receiving messages
def on_message(client, userdata, msg):
    while isAlarm:
        time.sleep(30)
    print(msg.payload)
    output = str(msg.payload).strip()
    x = "b'osayamen'"

    if output == x:
        for i in range(20):
            print(i)

    # output = msg.payload
    # if(is_json(msg.payload)):
    # output = json.dumps(json.loads(msg.payload), indent=4)

    # print(f"{msg.topic}: \n {output}")


condition_obj = threading.Condition()
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# set will message, when the Raspberry Pi is powered off,
# or the network is interrupted abnormally, it will send will message to other clients
client.will_set('raspberry/status', b'{"status": "Off"}')

# create connection, the three parameters are broker address, broker port number, and keep-alive time respectively
client.connect("broker.emqx.io", 1883, 60)

# set the network loop blocking, it will not actively end the program before calling disconnect() or the program crash
# client.loop_forever()

client.loop_start()

# should be infinite loop
while 1:
    time.sleep(20)
    check_time()
