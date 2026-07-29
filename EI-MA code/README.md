# EI-MA 核心算法代码（示意实现）

对应论文：**EI-MA: Expected Improvement-driven Graph-Constrained Multi-Agent** 数据增强框架。

> 本目录为**算法主干示意代码**，便于对照论文 Method 理解实现方式。  
> **不保证端到端可跑通**（无完整数据管线 / StarE 官方实现 / AnyBURL 缓存构建）。

## 模块对应关系

| 文件 | 论文内容 |
|------|----------|
| `graph/subject_graph.py` | 主体图 G_A、PPMI、d_A≤2 可配对 |
| `models/retrieval.py` | 局部证据检索 g、α、r |
| `models/communication.py` | 门控通信 ρ、消息 m、交互 z |
| `models/operators.py` | Skip / Recon / Struct-Pos / Hard-Neg |
| `training/ei_label.py` | 一阶 EI 估计 |
| `models/selector.py` | 共享选择器 π + ranking loss |
| `models/eima.py` | 一次训练更新的串联 |
| `training/losses.py` | 总损失 L = L_KGE + λ_g L_gen + λ_π L_π |
| `training/curriculum.py` | 三阶段课程学习 |
| `config.py` | 与论文 Table 一致的核心超参 |

## 推荐阅读顺序

1. `config.py`
2. `graph/subject_graph.py`
3. `models/retrieval.py` + `communication.py`
4. `models/operators.py`
5. `training/ei_label.py` + `models/selector.py`
6. `models/eima.py` + `training/train_step.py`
