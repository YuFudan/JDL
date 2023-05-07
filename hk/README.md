# 京东物流北京鸿坤营业部

## 文件介绍

- /orig_data: 营业部的路区、楼栋、路网、订单、轨迹等数据 (未实际放到GitLab上)
- get_regions.py: 读取处理路区数据 (每个路区为1个小哥负责的片区, 所有路区构成营业部的业务范围)
- get_road_network.py: 读取处理路网数据
- get_buildings.py: 读取处理楼栋数据
- read_order_traj.py: 读取订单、轨迹数据, 得到每小哥每天的轨迹与完成的订单
- get_wave_data.py: 处理订单、轨迹数据, 去噪、清洗、挖掘, 识别小哥一天中每"波"的数据 (携带一车货物到路区作业至送完为一波)
- constants_all.py: 定义通用常量/代码参数/工具函数/基础数据文件, 供其它代码import使用。包含几个部分的常量:
  - constants_file.py
  - constants_function.py
  - constants_spatial.py
  - ../constants_top.py
- params_eval.py: 控制主目录下一些通用算法代码作用于该营业部时的运行参数

## 写代码/跑代码顺序

### 下载原始数据

orig_data里应该放的文件, [在此下载](https://cloud.tsinghua.edu.cn/f/23608c8a5a3f42579fad/?dl=1)

### 写代码

1. find_center_and_station.ipynb 可视化路区和快递员轨迹, 决定中心坐标和仓库坐标
2. constants_spatial.py 填写上述坐标
3. get_regions.py
4. get_buildings.py
5. get_road_network.py
6. constants_file.py
7. constants_functions.py
8. constants_all.py
9. read_order_traj.py
10. get_wave_data.py
11. params_eval.py

### 跑代码

1. 运行evaluate.py中的data_prepare(), 跑出eval_data
2. seq_model_nn.py 训练好模型
3. 将训练好的模型路径填到params_eval.py
4. auto_bayes_tune.py
5. evaluate.py
