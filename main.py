import json
import time
import paho.mqtt.client as mqtt
import re
import os
import mysql.connector

from time import strftime
from pygame import mixer
from dotenv import load_dotenv, find_dotenv
from mysql.connector import errorcode

load_dotenv(find_dotenv())
db_config = {
    "user": os.getenv('DB_USER'),
    "password": os.getenv('PASSWORD'),
    "host": os.getenv('HOST'),
    "port": os.getenv('PORT'),
    "ssl_ca": os.getenv('SSL_CA'),
    "database": os.getenv('DATABASE'),
    "ssl_disabled": False
}

is_alarm_on = False
alarm_h = 15
alarm_m = 45

set_alarm_topic = 'raspberry/alarmclock/set_alarm'
retrieve_data_topic = 'raspberry/alarmclock/retrieve_data'
input_topics = [set_alarm_topic, retrieve_data_topic]

output_topic = 'raspberry/alarmclock/status'
mixer.init()
alarm_sound_file_path = os.getenv('ALARM_SOUND_FILE_PATH')
sound = mixer.Sound(alarm_sound_file_path)

# The below strings are sample JSON outputs for user queries.
# They may be hard to read inline, so use a JSON editor or view the output of the 'retrieved_data' variable.
time_query_output = {"10/4/2022 12:30": "80 secs", "10/4/2022 11:30": "50 secs"}
t_and_h_query_output = {"10/4/2022 12:30": {"Temp": "30 *F", "Humidity": "50%"}}


def on_connect(mqttclient, userdata, flags, rc):
    for topic in input_topics:
        mqttclient.subscribe(topic)


def on_message(mqttclient, userdata, msg):
    while is_alarm_on:  # Will not process request when alarm is ringing.
        time.sleep(0.5)
    p = str(msg.payload.decode("utf-8"))
    if msg.topic == set_alarm_topic:
        # TODO implement set alarm functionality
        mqttclient.publish(output_topic, payload="Alarm Set!", qos=0, retain=False)

    elif msg.topic == retrieve_data_topic:
        # TODO integrate retrieve data functionality
        retrieved_data = "Invalid Query"
        if re.match('^.*wakeupTime.*', p):
            retrieved_data = json.dumps(time_query_output, indent=2)
        elif re.match('^.*t&h.*', p):
            retrieved_data = json.dumps(t_and_h_query_output, indent=2)

        mqttclient.publish(output_topic, payload=retrieved_data, qos=0, retain=False)
    # TODO


def check_alarm():
    while 1:
        current_h = int(strftime("%H"))
        current_m = int(strftime("%M"))

        if (alarm_h == current_h) and (alarm_m == current_m):
            global is_alarm_on
            is_alarm_on = True
            sound.play()
            client.publish(output_topic, payload="ALARM ON!", qos=0, retain=False)
            time.sleep(5)
            # TODO face processing
            is_alarm_on = False
            mixer.pause()
            time.sleep(60)

        time.sleep(10)


def init_database():
    try:
        conn = mysql.connector.connect(**db_config)
        print("Connection established Successfully!")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with the user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        return conn.cursor()


if __name__ == "__main__":
    # establish db connection
    db_cursor = init_database()

    broker_address = "broker.emqx.io"
    broker_port_number = 1883
    broker_keep_alive_time = 60

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.will_set('raspberry/alarmclock/system_status', '{"status": "Off"}')
    client.connect(broker_address, broker_port_number, broker_keep_alive_time)

    client.loop_start()

    check_alarm()
