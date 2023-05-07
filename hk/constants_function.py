import time

def date2num(date):
    """
    例如数据中最早1天是2月1日, 可以设置ref为1月31日, 这样2月1日得到的数就是1, 比较好看
    """
    ref = time.mktime(time.strptime("2023-01-31 00:00:00", "%Y-%m-%d %H:%M:%S"))
    t = time.mktime(time.strptime(date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))
    return int((t - ref) / 86400)
