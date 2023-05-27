create table facial_data
(
    id        bigint unsigned auto_increment
        primary key,
    username  varchar(50) not null,
    encodings mediumblob  not null,
    constraint id
        unique (id)
);

create table temp_and_humidity
(
    id          bigint unsigned auto_increment
        primary key,
    recorded_on varchar(20) null,
    temp_F      int         null,
    humidity    int         null,
    constraint id
        unique (id)
);

create table wake_up_durations
(
    id                         bigint unsigned auto_increment
        primary key,
    username                   varchar(50) not null,
    alarm_date                 varchar(20) not null,
    wake_up_duration           varchar(10) not null,
    completed_face_recognition tinyint(1)  not null,
    constraint id
        unique (id)
);