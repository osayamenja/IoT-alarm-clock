import json
import time
import paho.mqtt.client as mqtt
import re
import os
import mysql.connector
import cv2
import vlc
import face_recognition
import pickle
import imutils
import datetime
import threading
import board
import adafruit_dht
import psutil

from gtts import gTTS
from picamera import PiCamera
from imutils import paths
from imutils.video import VideoStream
from picamera.array import PiRGBArray
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
alarm_day = None
user_name = None
wake_up_duration = None
awaiting_registration = False

v = vlc.Instance()
p = v.media_player_new()
set_alarm_topic = 'raspberry/alarmclock/set_alarm'
retrieve_data_topic = 'raspberry/alarmclock/retrieve_data'
register_user_topic = 'raspberry/alarmclock/register_user'
input_topics = [set_alarm_topic, retrieve_data_topic, register_user_topic]

output_topic = 'raspberry/alarmclock/status'
mixer.init()
alarm_sound_file_path = os.getenv('ALARM_SOUND_FILE_PATH')
sound = mixer.Sound(alarm_sound_file_path)
encoding_data_file_path = os.getenv('ENCODING_DATA_FILE_PATH')
image_data_file_path = os.getenv('IMAGE_DATA_FILE_PATH')
max_wake_up_seconds = 30

tts_file_path = os.getenv('TTS_FILE_PATH')

facial_data_table_name = os.getenv('FACIAL_DATA_TABLE_NAME')
wake_up_duration_table_name = os.getenv('WAKE_DURATION_TABLE_NAME')
t_and_h_table_name = os.getenv('T_AND_H_TABLE_NAME')

# The below strings are sample JSON outputs for user queries.
# They may be hard to read inline, so use a JSON editor or view the output of the 'retrieved_data' variable.
time_query_output = {"10/4/2022 12:30": "80 secs", "10/4/2022 11:30": "50 secs"}
t_and_h_query_output = {"10/4/2022 12:30": {"Temp": "30 *F", "Humidity": "50%"}}


# Format: 10/21/2022 12:00 AM
def get_formatted_timestamp(input_datetime):
    t = input_datetime.strftime("%H:%M")
    result = datetime.datetime.strptime(t, "%H:%M").strftime("%I:%M %p")
    return next_hour.strftime("%m/%d/%Y") + " " + result


def get_next_hour_date_time():
    delta = datetime.timedelta(hours=1)
    now = datetime.datetime.now()
    return (now + delta).replace(microsecond=0, second=0, minute=0)


# job to upload ambient temperature and humidity hourly to a database 
def temp_and_humidity_job(dht_device):
    next_hour = get_next_hour_date_time()
    while True:
        if datetime.datetime.now() >= next_hour:
            complete_upload_job = False

            # Reading sensor data occasionally causes errors hence the loop.
            while not complete_upload_job:
                try:
                    temperature_c = dht_device.temperature
                    temperature_f = temperature_c * (9 / 5) + 32
                    humidity = dht_device.humidity

                    upload_t_and_h_to_db(get_formatted_timestamp(next_hour), temperature_f, humidity)
                    complete_upload_job = True

                except RuntimeError as error:
                    # Errors happen fairly often, DHT's are hard to read, just keep going
                    print(error.args[0])
                    time.sleep(2.0)
                    continue
                except Exception as error:
                    dhtDevice.exit()
                    raise error

                time.sleep(2.0)

            next_hour = get_next_hour_date_time()
        else:
            time.sleep(1)


def upload_t_and_h_to_db(recorded_datetime, temperature_f, humidity):
    cursor = db_conn.cursor()
    q = "INSERT INTO {} (recorded_on, temperature_F, humidity) VALUES (%s, %s, %s)"
    q = q.format(t_and_h_table_name)
    cursor.execute(q, (recorded_datetime, temperature_f, humidity))
    db_conn.commit()
    cursor.close()


# 11:00 PM -> 23:00 
def extract_hour_and_minute(input_time):
    if re.match('\\d{1,2}:\\d{2}\\s[AP]M', input_time):
        parsed_time = re.split(':|\\s', input_time)
        h = int(parsed_time[0])
        if parsed_time[2] == 'PM':
            h += 12
        elif h == 12:
            h = 0
        return h, parsed_time[1], True
    else:
        return None, None, False


def get_date():
    return datetime.datetime.now().strftime("%m/%d/%Y")


# 23:00 -> 11:00 PM
def get_12_hour_date_time(input_time):
    twelve_hr_time = datetime.datetime.strptime(input_time, "%H:%M").strftime("%I:%M %p")
    return str(alarm_day) + " " + twelve_hr_time


def speak_text(input_text):
    tts = gTTS(input_text)
    tts.save(tts_file_path)
    p.set_media(v.media_new(tts_file_path))
    p.play()
    time.sleep(5)


def is_user_registered(input_username):
    cursor = db_conn.cursor()
    cursor.execute("SELECT username FROM {} WHERE username = %s".format(facial_data_table_name), (input_username,))
    result = len(cursor.fetchall())
    db_conn.commit()
    cursor.close()
    return result > 0


def convert_to_binary_data(filename):
    with open(filename, 'rb') as file:
        b = file.read()
    return b


def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)


def write_encodings_file_from_db_to_disk(input_username):
    cursor = db_conn.cursor()
    cursor.execute("SELECT encodings FROM {} WHERE username = %s".format(facial_data_table_name), (input_username,))
    record = cursor.fetchall()  # There should only be one encoding per user.
    write_file(record[0][0], encoding_data_file_path)
    db_conn.commit()
    cursor.close()


def insert_wake_up_time(input_username, wakeup_duration, completed_face_detection, input_time):
    cursor = db_conn.cursor()
    q = "INSERT INTO {} (username, alarm_date, wake_up_duration, completed_face_recognition) VALUES (%s, %s, %s, %s)"
    q = q.format(wake_up_duration_table_name)
    cursor.execute(q, (input_username, get_12_hour_date_time(input_time), wakeup_duration, completed_face_detection))
    db_conn.commit()
    cursor.close()


def upload_encodings_to_db(input_username):
    cursor = db_conn.cursor()
    insert_query = "INSERT INTO {} (username, encodings) VALUES (%s, %s)".format(facial_data_table_name)
    binary_encoding = convert_to_binary_data(encoding_data_file_path)
    cursor.execute(insert_query, (input_username, binary_encoding))
    db_conn.commit()
    cursor.close()


def persist_images_to_disk():
    cam = PiCamera()
    cam.resolution = (512, 304)
    cam.framerate = 10
    raw_capture = PiRGBArray(cam, size=(512, 304))
    img_counter = 0

    speak_text("Please look at the camera and stay still for five seconds. Photo capture will begin in five seconds")
    time.sleep(5)

    for frame in cam.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        raw_capture.truncate(0)

        img_name = image_data_file_path + "image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, image)
        print("{} written!".format(img_name))
        img_counter += 1
        time.sleep(0.5)
        if img_counter == 12:
            break

    speak_text("Photo capture is complete!")
    cv2.destroyAllWindows()


# Source: https://core-electronics.com.au/guides/face-identify-raspberry-pi/#What
def train_model():
    image_paths = list(paths.list_images(image_data_file_path))
    known_encodings = []

    for (i, imagePath) in enumerate(image_paths):
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)

    print("[INFO] persisting encodings locally...")
    data = {"encodings": known_encodings}
    write_file(pickle.dumps(data), encoding_data_file_path)


# Source: https://core-electronics.com.au/guides/face-identify-raspberry-pi/#What
def perform_facial_recognition(user_wake_up_time, alarm_timeout):
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encoding_data_file_path, "rb").read())
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    current_time = time.time()
    wake_up_end = current_time + user_wake_up_time
    alarm_end = current_time + alarm_timeout
    current_time = time.time()
    found_user = False
    speak_text("Starting facial recognition...")

    while current_time < wake_up_end and current_time < alarm_end:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)
        # compute the facial embeddings for the user's face bounding box
        facial_encodings = face_recognition.face_encodings(frame, boxes)

        if len(facial_encodings) == 0:
            found_user = False
            speak_text("User's Face is not detected. Please show your face to the camera")
            wake_up_end = current_time + user_wake_up_time
            time.sleep(2)

        else:
            encoding = face_recognition.face_encodings(frame, boxes)[0]

            matches = face_recognition.compare_faces(data["encodings"], encoding)

            # update time
            current_time = time.time()
            if True in matches:
                if not found_user:
                    found_user = True
                    speak_text("Found face. Please stay still and keep your eyes open")
                    time.sleep(2)

            else:
                found_user = False
                speak_text("User's Face is not detected. Please show your face to the camera")
                wake_up_end = current_time + user_wake_up_time
                time.sleep(2)

    vs.stop()
    if current_time < alarm_end:
        return True
    else:
        return False


def register_user(input_username):
    persist_images_to_disk()
    train_model()
    upload_encodings_to_db(input_username)


def on_connect(mqttclient, userdata, flags, rc):
    for topic in input_topics:
        mqttclient.subscribe(topic)


def configure_alarm(mqttclient, parsed_input, input_username):
    input_time = parsed_input[1].strip()
    global alarm_m
    global alarm_h
    global wake_up_duration
    global user_name
    global alarm_day
    h, m, is_time_valid = extract_hour_and_minute(input_time)
    w = parsed_input[2].strip()

    if is_time_valid and w.isnumeric() and int(w) <= max_wake_up_seconds:
        alarm_h = h
        alarm_m = m
        alarm_day = get_date()
        wake_up_duration = int(w)
        user_name = input_username
        mqttclient.publish(output_topic, payload="Alarm set!", qos=0, retain=False)

    elif not is_time_valid:
        mqttclient.publish(output_topic, payload="Invalid input, please reenter request", qos=0, retain=False)

    else:
        response = f"Invalid wakeup time, range is 0 to {max_wake_up_seconds}"
        mqttclient.publish(output_topic, payload=response, qos=0, retain=False)


def set_alarm(mqttclient, user_input):
    parsed_input = user_input.split(',')
    username = parsed_input[0].strip()
    if is_user_registered(username):
        mqttclient.publish(output_topic, payload="Username Verified!", qos=0, retain=False)
        mqttclient.publish(output_topic, payload="Setting Alarm...", qos=0, retain=False)
        write_encodings_file_from_db_to_disk(username)
        configure_alarm(mqttclient, parsed_input, username)
    else:
        global awaiting_registration
        awaiting_registration = True
        mqttclient.publish(output_topic, payload="Username not registered", qos=0, retain=False)
        response = "Register at 'register_user' then re-send request"
        mqttclient.publish(output_topic, payload=response, qos=0, retain=False)


def on_message(mqttclient, userdata, msg):
    while is_alarm_on:  # Will defer processing any request when alarm is ringing.
        time.sleep(0.5)
    mqtt_payload = str(msg.payload.decode("utf-8"))
    if msg.topic == set_alarm_topic:
        set_alarm(mqttclient, mqtt_payload)
    elif msg.topic == register_user_topic:
        global awaiting_registration
        if awaiting_registration:
            parsed_input = mqtt_payload.split(',')
            mqttclient.publish(output_topic, payload="Starting registration...", qos=0, retain=False)
            register_user(parsed_input[0])
            mqttclient.publish(output_topic, payload="Registration complete!", qos=0, retain=False)
            mqttclient.publish(output_topic, payload="Re-send request to set alarm", qos=0, retain=False)
            awaiting_registration = False
        else:
            mqttclient.publish(output_topic, payload="Wrong input to 'register_user'", qos=0, retain=False)
    elif msg.topic == retrieve_data_topic:
        # TODO retrieve_data
        retrieved_data = "Invalid Query"
        if re.match('^.*wakeupTime.*', mqtt_payload):
            retrieved_data = json.dumps(time_query_output, indent=2)
        elif re.match('^.*t&h.*', mqtt_payload):
            retrieved_data = json.dumps(t_and_h_query_output, indent=2)

        mqttclient.publish(output_topic, payload=retrieved_data, qos=0, retain=False)


def check_alarm():
    global is_alarm_on
    global wake_up_duration
    global max_wake_up_seconds
    global alarm_m
    global alarm_h
    global user_name
    global alarm_day
    while True:
        current_h = int(strftime("%H"))
        current_m = int(strftime("%M"))

        if (alarm_h == current_h) and (alarm_m == current_m):

            is_alarm_on = True
            sound.play(-1)
            client.publish(output_topic, payload="ALARM ON!", qos=0, retain=False)
            time.sleep(5)
            complete_face_detection = perform_facial_recognition(wake_up_duration, (max_wake_up_seconds + 20))
            mixer.stop()
            alarm_report = "Successfully Completed user's facial recognition."

            if not complete_face_detection:
                alarm_report = "Unsuccessful in recognizing the user's face before alarm timeout."

            speak_text(alarm_report)
            alarm_time = str(alarm_h) + ":" + str(alarm_m)
            insert_wake_up_time(user_name, wake_up_duration, complete_face_detection, alarm_time)

            # flush alarm data
            alarm_m = None
            alarm_h = None
            wake_up_duration = None
            user_name = None
            alarm_day = None

            is_alarm_on = False


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


def init_sensor():
    for proc in psutil.process_iter():
        if proc.name() == 'libgpiod_pulsein' or proc.name() == 'libgpiod_pulsei':
            proc.kill()
    return adafruit_dht.DHT11(board.D23)


if __name__ == "__main__":
    dht_sensor = init_sensor()
    t_and_h_upload_worker = threading.Thread(target=temp_and_humidity_job, args=(dht_sensor,), daemon=True)
    print("Starting temperature and humidity upload worker...")
    t_and_h_upload_worker.start()

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
