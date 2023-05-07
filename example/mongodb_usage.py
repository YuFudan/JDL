"""
MongoDB使用示例
"""
import pymongo

orders = [
    {
        "id": 0,
        "building": "1号楼",
        "floor": 2,
        "unit": 1,
        "start_time": 32400.0,
        "end_time": 33000.0
    }, 
    {
        "id": 1,
        "building": "2号楼",
        "floor": 2,
        "unit": 1,
        "start_time": 32400.0,
        "end_time": 33000.0
    }, 
    {
        "id": 2,
        "building": "3号楼",
        "floor": 2,
        "unit": 1,
        "start_time": 32400.0,
        "end_time": 33000.0
    }
]

client = pymongo.MongoClient("")
db_name = "jd_digital_twin"  # database
col_name = "test"    # collection
col = client[db_name][col_name]

col.drop()  # 清空collection(若不清空, 则会在现有内容后添加)
col.insert_many(orders, ordered=False)  # 写入collection

orders = list(col.find({}, {"_id":False}))  # 读collection
print(orders)
