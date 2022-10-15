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

db_conn = None
is_alarm_on = False
alarm_h = None
alarm_m = None
wake_up_time = None

set_alarm_topic = 'raspberry/alarmclock/set_alarm'
retrieve_data_topic = 'raspberry/alarmclock/retrieve_data'
register_user_topic = 'raspberry/alarmclock/register_user'
input_topics = [set_alarm_topic, retrieve_data_topic, register_user_topic]

output_topic = 'raspberry/alarmclock/status'
mixer.init()
alarm_sound_file_path = os.getenv('ALARM_SOUND_FILE_PATH')
sound = mixer.Sound(alarm_sound_file_path)
facial_encoding_file_path = os.getenv('ENCODING_FILE_PATH')
max_wake_up_seconds = 30

# The below strings are sample JSON outputs for user queries.
# They may be hard to read inline, so use a JSON editor or view the output of the 'retrieved_data' variable.
time_query_output = {"10/4/2022 12:30": "80 secs", "10/4/2022 11:30": "50 secs"}
t_and_h_query_output = {"10/4/2022 12:30": {"Temp": "30 *F", "Humidity": "50%"}}


def extract_hour_and_minute(input_time):
    return 12, 45, True
    # TODO for team member working on set_alarm


def is_user_registered(input_username):
    cursor = db_conn.cursor()
    cursor.execute("SELECT username FROM facial_data WHERE username = %s", (input_username,))
    return len(cursor.fetchall()) > 0


def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)


def write_encodings_file_from_db_to_disk(input_username):
    cursor = db_conn.cursor()
    cursor.execute("SELECT encodings FROM facial_data WHERE username = %s", (input_username,))
    record = cursor.fetchone()  # There should only be one encoding per user.
    write_file(record[0][1], facial_encoding_file_path)


def register_user(input_username):
    print("working...")


def on_connect(mqttclient, userdata, flags, rc):
    for topic in input_topics:
        mqttclient.subscribe(topic)


def configure_alarm(mqttclient, parsed_input):
    input_time = parsed_input[1].strip()
    global alarm_m
    global alarm_h
    global wake_up_time
    alarm_h, alarm_m, is_time_valid = extract_hour_and_minute(input_time)
    wake_up_time = parsed_input[2].strip()

    if is_time_valid and wake_up_time.isnumeric() and wake_up_time <= max_wake_up_seconds:
        mqttclient.publish(output_topic, payload="Alarm set!", qos=0, retain=False)

    elif not is_time_valid:
        mqttclient.publish(output_topic, payload="Invalid input, please reenter request", qos=0, retain=False)

    else:
        response = f"Invalid wakeup time, range is 0 to {max_wake_up_seconds}"
        mqttclient.publish(output_topic, payload=response, qos=0, retain=False)


def set_alarm(mqttclient, user_input):
    parsed_input = user_input.split(',')
    username = parsed_input[0]
    if is_user_registered(username):
        mqttclient.publish(output_topic, payload="Username Verified!", qos=0, retain=False)
        mqttclient.publish(output_topic, payload="Setting Alarm...", qos=0, retain=False)
        write_encodings_file_from_db_to_disk(username)
        configure_alarm(mqttclient, parsed_input)
    else:
        response = "Username not registered\n" \
                   "Register Username?"
        mqttclient.publish(output_topic, payload=response, qos=0, retain=False)


def on_message(mqttclient, userdata, msg):
    while is_alarm_on:  # Will not process request when alarm is ringing.
        time.sleep(0.5)
    p = str(msg.payload.decode("utf-8"))
    if msg.topic == set_alarm_topic:
        set_alarm(mqttclient, p)

    elif msg.topic == register_user_topic:
        parsed_input = p.split(',')
        # register_user()
        configure_alarm(mqttclient, parsed_input)

    elif msg.topic == retrieve_data_topic:
        # TODO retrieve_data
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
        return conn


if __name__ == "__main__":
    # establish db connection
    db_conn = init_database()

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
