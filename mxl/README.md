# 京东物流北京漫香林营业部

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
