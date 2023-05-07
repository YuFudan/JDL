# 京东物流终端仿真项目——快递员行为还原及模拟

已删去业务数据, 无法运行完整系统, 仅保留部分模块核心代码

## 主目录下文件说明

```text
不同物流终端营业部之间通用的核心代码, 例如行为还原算法、模拟模型、生成前端DEMO数据相关代码等
```

- gen_sim_actions.py: 快递员模拟模型
  - ABM模型, 通过调整一组快递员个性化参数, 可使得模型拟合小哥的轨迹和订单数据
  - 直接主动运行gen_sim_actions.py时, 主函数中的相关代码久未维护, 已不适用, 被auto_bayes_tune调用时可正常使用
- seq_model_nn.py: 一个学习快递员派送序列(送完一栋楼以后下一栋楼去哪)的NN模型
- auto_bayes_tune.py: 模型参数的学习
  - 给定一定量的轨迹和订单数据作为训练集, 按照evaluate.py中的误差计算方式定义loss, 对模型参数进行搜索
- evaluate.py: 模型的评估
  - 给定真实的订单和轨迹, 以及模拟模型输出的actions, 定义合适的指标, 分别计算真值指标和模拟指标, 评估模拟误差
- lkh.py: 求解VRP问题的LKH算法, 其核心为C++代码, 此为python入口
  - 供gen_sim_actions.py调用
  - 使用前需要先在命令行运行指令"bash compile"对C++代码进行编译 (将生成一个"LKH"文件)
- constants_top.py: 定义一些与营业部无关的通用常量
- courier-dtrec/: 新版行为还原算法 (KDD2023)
- 其它历史代码
  - gen_actions.py: 最早期开发的快递员行为还原代码, 久未维护了
    - 但当时对于订单、轨迹数据中的误差还没有清晰地认识, 并未很好地解决数据误差对还原带来的困难
  - action2status.py: 生成前端DEMO数据
    - 将gen_actions.py或gen_sim_actions.py输出的actions解析为前端DEMO所需的数据
  - replan_absent_orders.py
    - 生成小哥请假DEMO下的gen_sim_actions的输入
  - replan_partition_orders.py
    - 生成路区重新划分DEMO下的gen_sim_actions的输入

## 各营业部目录, 包括mxl/和hk/

见各营业部目录下的README
