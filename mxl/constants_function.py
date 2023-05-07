import time

def date2num(date):
    ref = time.mktime(time.strptime("2022-07-31 00:00:00", "%Y-%m-%d %H:%M:%S"))
    t = time.mktime(time.strptime(date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))
    return int((t - ref) / 86400)
