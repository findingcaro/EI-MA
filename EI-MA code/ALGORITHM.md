# EI-MA 实现方式说明（对照论文）

## 1. 总体闭环

```
训练主体图 G_A（PPMI + mutual top-K, d_A≤2）
        ↓
采样可配对 (i,j) + 共享锚点事实 q0
        ↓
局部检索 → u_ij
        ↓
EI 标签（probe 梯度 vs 四算子梯度）→ 训练选择器 L_π
        ↓
从 π 采样动作 a（离散路由 stop-gradient）
        ↓
若 a≠Skip：门控通信 → z_ij → L_dec + L_aug(a)
        ↓
总损失 L = L_KGE + λ_g L_gen + λ_π L_π
        ↓
测试：仅用融合后的静态嵌入 e+ 做 StarE 打分（无选择器/增强）
```

## 2. 关键设计点（实现时务必对齐论文）

1. **同一决策共享状态**：四个动作的 EI 标签必须在同一 `q0`、同一 backbone 状态下计算，保证相对可比。  
2. **Skip 的 EI=0**：动作为 Skip 时增广梯度为 0。  
3. **选择器只吃 ranking 监督**：对 `u_ij` 使用 stop-gradient；不对离散动作做 policy-gradient。  
4. **候选无泄漏**：`C_un = C \ T(s,r)`，且构图/规则/统计只用训练集。  
5. **Hard-Neg 置信权重**：`w(δ)=sg[σ(βδ)]`，减轻假阴性风险。  
6. **Struct-Pos**：AnyBURL 风格离线缓存可替换；空集回退 Recon。  
7. **课程学习**：先 L_KGE → 强制 Recon → 全四动作 + 周期性刷新 EI。

## 3. 你需要自行补齐才能真正训练的部分

- 真实 **StarE** 编码器与 qualifier 消息传递  
- 数据集加载（WikiPeople / WD50K / JW44K）与 filtered 评估  
- AnyBURL 规则挖掘与置信度缓存  
- 批量化的候选打分（当前 Hard-Neg 为示意循环）  
- 分布式 / 多卡、日志与 checkpoint  

## 4. 文件入口

阅读顺序见 `README.md`。核心串联类：`models/eima.py` 中的 `EIMA.forward_pair_update`。
