ID_COLUMNS = ["unit_id","cycle"]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]

SENSOR_COLUMNS = [
    f"sensor_{i}" for i in range(1,22)
]

FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS

ALL_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS

TARGET_COLUMNS = "RUL"