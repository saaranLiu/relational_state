# Relational-State 评测结果总览

本文件汇总当前仓库中最新评测输出（`evaluation/outputs/*`）对应的主要结果，用于论文写作与后续微调设计参考。


## 理论基础公式（Langtry）

主任务的核心决策函数为：

`x_i^* = F + alpha_i * R_i`, 其中 `R_i = Σ_j g_ij x_j`。

等价写法：`x_i^* = F + alpha_i * Σ_j g_ij x_j`。

含义是：个体先有私有基线 `F`，再按 closeness 权重 `g_ij` 聚合他人行动形成社会参照，并由社会敏感度 `alpha_i` 放大或缩小该参照影响。

## 1) 评测任务说明

- `eval_A`: 规则识别（是否选择 theory-based 规则）
- `eval_B`: 比较静态方向判断（pairwise, 判断单调性）
- `placebo_test`: 非社会比较场景（ground truh为私有基线）
- `ood_social`: OOD 社交匹配任务（基于 `b/c` 匹配稳定性）
- `ood_career`: OOD 职业排序任务（绝对薪资 vs 相对地位阈值权衡）

### 1.0) 各任务样本数（每个模型一致）

| Split | Evaluation units |
|---|---:|
| `eval_A` | 1,512 |
| `eval_B` | 280 pairs |
| `placebo_test` | 504 |
| `ood_social` | 576 |
| `ood_career` | 576 |

说明：`eval_B` 的单位是成对比较样本 pair，其余任务是单题样本数。

## 1.1) MCQ 设计规则（含选项语义）

### A) `eval_A` / `placebo_test`（四选一，多规则对比）

- 形式：每题展示 4 个候选规则描述（从 A-H 规则池中抽取，含 1 个 gold + 3 个 distractor）
- 目标：不是考算术计算，而是考模型是否选择了**正确行为机制**
- 采样说明：**先固定 ground truth 选项，再从其余规则中均匀抽取 3 个 distractor**（并做可辨识性过滤）

规则池（A-H）语义如下：

- `A_peer_weighted`：`x = F + alpha * Σ(g_ij * x_j)`  
  含义：Langtry 主规则，按亲近权重聚合同伴影响（主任务 gold）
- `B_top_anchor`：`x = F + alpha * x_top`  
  含义：只看最亲近/最高锚点同伴
- `C_uniform_avg`：`x = F + alpha * mean(x_j)`  
  含义：把同伴等权平均，忽略 closeness 权重
- `D_pure_private`：`x = F`  
  含义：完全按私有基线，不受同伴影响（placebo gold）
- `E_closest_mimicry`：`x = x_top`  
  含义：直接模仿最亲近同伴
- `F_median_anchor`：`x = F + alpha * median(x_j)`  
  含义：以中位同伴作锚
- `G_counter_conformist`：`x = F - alpha * Σ(g_ij * x_j)`  
  含义：反从众（与社会参照反向）
- `H_equal_mix`：`x = 0.5*F + 0.5*Σ(g_ij * x_j)`  
  含义：私有基线与社会参照等权混合

gold 规则说明：

- `eval_A`（positional 任务）gold = `A_peer_weighted`
- `placebo_test`（non-positional 任务）gold = `D_pure_private`

### B) `eval_B`（二选一，比较静态）

- 形式：给定同一场景的 A/B 两个版本，问哪一侧最终承诺更高
- 选项语义：
  - `A`：Scenario A 更高
  - `B`：Scenario B 更高
- 扰动类型（仅变动一维）：
  - `alpha_i_up`
  - `F_up`
  - `ref_sum_up`
  - `top_weight_up`
  - `peer_action_up`

### C) `ood_social`（二选一，稳定匹配）

- 形式：两个候选对象（二选一）
- 选项语义：
  - `A` / `B`：对应两个 candidate
- gold 规则：选择与主角 `b/c` 比率更匹配的一方（稳定匹配准则）
- 说明：这里的 OOD 不仅是场景语义变化，也包含决策函数形式变化，从主任务的加权参照函数切换为 `b/c` 匹配函数。

### D) `ood_career`（二选一，岗位选择）

- 形式：两个 firm 方案（二选一）
- 选项语义：
  - `A` / `B`：对应两个 firm（高薪大池 vs 低薪小池等配置）
- gold 规则：满足 Langtry 阈值比较  
  `x_S_H - x_S_L >= alpha_2i * (x̄_H - x̄_L)` 时选 `H`，否则选 `L`
- 说明：这里的 OOD 不仅是场景语义变化，也包含决策函数形式变化，从主任务规则切换为阈值比较函数。

### E) 采样规则（简版）

- **场景采样**：按 domain/family 划分 train/test，避免同一场景词面泄漏。  
- **潜变量采样**：主任务按 `alpha × dispersion × skew` 三维网格采样；包含 held-out cells 测泛化。  
- **选项采样**：每题固定 1 个 gold，其他选项从候选规则中均匀抽取（再通过 margin/tau 过滤，保证可区分）。  
- **均衡约束**：控制 gold letter、distractor rule 使用频次与子桶覆盖，减少位置偏差和类别偏斜。  
- **OOD 采样**：  
  - `ood_social`：按 `b/c` 匹配距离桶（close/mid/far）均衡；  
  - `ood_career`：按 `alpha_2i` 桶、目标 firm、gold letter 联合均衡。

## 1.2) 评测任务的设计

将结构能力拆成互补子能力，回答一个核心问题：  **模型是否真的学会了社会比较机制，而不是用表面启发式蒙对。**

### A) `eval_A` 的意义：机制识别能力 （主要评估任务）

- 检验模型是否能区分 `A_peer_weighted` 与常见替代规则（平均、最近同伴、私有基线等）。
- 如果 `eval_A` 低，即使其他任务不错，也说明模型并未真正掌握 Langtry 的结构。
- 对微调价值：可直接把错因映射为训练对比样本（gold 机制 vs shortcut 机制）。

### B) `eval_B` 的意义：比较静态敏感性

- 检验模型是否能对单变量扰动做正确方向判断（`alpha`、`F`、`ref_sum` 等）。
- 它衡量方向感而非完整规则恢复，因此通常比 `eval_A` 容易。
- 对微调价值：帮助定位模型是机制不会，还是机制会但局部扰动不稳。

### C) `placebo_test` 的意义：机制门控与过度社会化抑制

- 在不该启用社会比较的场景中，检验模型能否回到 `D_pure_private`。
- 该任务专门识别过度社会化偏差，即看到同伴信息就误加社会项。
- 对微调价值：作为反例约束，避免模型把社会比较规则无差别套用到所有任务。

### D) `ood_social` 的意义：跨任务迁移（社交匹配）

- 检验模型能否从主任务迁移到不同语义表面下的 `b/c` 稳定匹配逻辑。
- 不再是同一题型重做，而是看结构知识是否可转用。该 OOD 同时考察场景分布偏移与决策函数偏移。
- 对微调价值：验证学到规则，而不是记住题型模板。

### E) `ood_career` 的意义：冲突权衡能力（绝对收益 vs 相对地位）

- 该任务故意制造高绝对薪资与高相对地位冲突，测试阈值比较。
- 该 OOD 同时考察场景分布偏移与决策函数偏移。
- 若接近随机，说明模型在多因素冲突决策时仍缺稳定结构策略。
- 对仿真价值：进入社会动态模拟前的关键压力测试。

### F) 组合解释

- `eval_A` 高 + `placebo_test` 高：说明模型既会用规则，也会在不该用时停用（有门控）。我们预期能够看到的微调效果是各任务提升的同时，placebo test尽量不降低。
- `eval_B` 高但 `eval_A` 低：说明仅有方向性启发式，不具备完整机制识别。
- `OOD` 高：说明结构能力可迁移；`OOD` 低：说明仍偏模板依赖。

因此，本评测体系的价值在于把是否具备社会比较推理能力拆解为：  
**规则识别、方向敏感、门控抑制、跨任务迁移、冲突权衡** 五个维度，并将错误直接转化为后续微调目标。

## 2) 各模型主结果（Accuracy, %）

来源：`analysis/model_performance_error_analysis/accuracy_by_model_task.csv`

| Model | eval_A | eval_B | placebo_test | ood_social | ood_career |
|---|---:|---:|---:|---:|---:|
| gpt4 | 🏆 **51.59** | 62.50 | 🏆 **96.03** | 🏆 **69.44** | 50.00 |
| ollama-qwen2-7b | 46.63 | 56.43 | 91.67 | 59.20 | 47.57 |
| ollama-llama3-1-latest | 41.73 | 61.79 | 75.20 | 61.98 | 50.35 |
| ollama-llama3-8b | 40.34 | 58.21 | 79.37 | 59.72 | 🏆 **51.74** |
| Qwen | 33.86 | 🏆 **70.00** | 59.72 | 65.97 | 49.48 |
| DeepSeek-R1-671B | 23.41 | 59.64 | 29.96 | 56.94 | 45.83 |

图例：🏆 表示该任务下最高准确率。

## 3) 任务级 best/worst 与区分度（spread）

来源：`analysis/model_performance_error_analysis/model_gap_by_task.csv`

- `eval_A`: best = **51.59** (`gpt4`), worst = 23.41, spread = **28.17**
- `eval_B`: best = **70.00** (`Qwen`), worst = 56.43, spread = 13.57
- `placebo_test`: best = **96.03** (`gpt4`), worst = 29.96, spread = **66.07**
- `ood_social`: best = **69.44** (`gpt4`), worst = 56.94, spread = 12.50
- `ood_career`: best = **51.74** (`ollama-llama3-8b`), worst = 45.83, spread = 5.90

结论：`eval_A` 依旧是核心结构能力的区分任务；`placebo_test` 在是否抑制不该有的社会比较上拉开了最大差距；`ood_career` 全体仍接近 chance 区间。

## 3.1) Eval-A 27-cell 参数含义与分层分析

Eval-A 的 27 cells 来自三维参数网格：`alpha × dispersion × skew`，每个维度三档（low/mid/high）。

参数含义：

- `alpha`：社会比较强度，即个体对社会参照 `R_i` 的敏感程度。  
  越高表示越容易被同伴参照拉动。
- `dispersion`：同伴权重 `g_ij` 的离散程度。  
  low 更接近均匀权重，high 更接近单一同伴主导。
- `skew`：同伴行动值 `x_j` 的偏态或拉开程度。  
  low 表示同伴行动接近、对比度弱；high 表示同伴差异明显、对比度强。

本次 27-cell 统计（见 notebook 新增 `Eval-A 27-Cell Analysis` 与导出 CSV）显示：

- **共同难点集中在 low-skew 相关 cell**，说明当同伴行动差异不明显时，模型更容易把加权聚合规则退化为捷径规则。  
- 在 `alpha` 中高且 `skew` 低的一些 cell 上，跨模型平均准确率更低，体现了高社会敏感场景下规则错配的放大效应。  
- 存在若干 **高 spread cell**（不同模型差异大），说明这些 cell 更能区分结构能力，而不是随机波动。

### 1.2.1 分模型参数统计（不用总体平均）

#### A) `alpha` 分桶准确率（%）

| Model | low | mid | high | best bucket |
|---|---:|---:|---:|---|
| gpt4 | 🏆 **63.89** | 49.40 | 41.47 | low |
| ollama-qwen2-7b | 🏆 **53.97** | 43.65 | 42.26 | low |
| ollama-llama3-1-latest | 🏆 **45.44** | 40.67 | 39.09 | low |
| ollama-llama3-8b | 🏆 **45.83** | 37.30 | 37.90 | low |
| Qwen | 🏆 **40.67** | 32.94 | 27.98 | low |
| DeepSeek-R1-671B | 🏆 **27.58** | 23.41 | 19.25 | low |

#### B) `dispersion` 分桶准确率（%）

| Model | low | mid | high | best bucket |
|---|---:|---:|---:|---|
| gpt4 | 🏆 **52.38** | 50.40 | 51.98 | low |
| ollama-qwen2-7b | 46.63 | 44.84 | 🏆 **48.41** | high |
| ollama-llama3-1-latest | 40.87 | 42.06 | 🏆 **42.26** | high |
| ollama-llama3-8b | 38.49 | 41.07 | 🏆 **41.47** | high |
| Qwen | 33.73 | 🏆 **37.90** | 29.96 | mid |
| DeepSeek-R1-671B | 23.61 | 🏆 **24.21** | 22.42 | mid |

#### C) `skew` 分桶准确率（%）

| Model | low | mid | high | best bucket |
|---|---:|---:|---:|---|
| gpt4 | 44.25 | 53.97 | 🏆 **56.55** | high |
| ollama-qwen2-7b | 41.07 | 46.83 | 🏆 **51.98** | high |
| ollama-llama3-1-latest | 36.90 | 42.26 | 🏆 **46.03** | high |
| ollama-llama3-8b | 30.16 | 42.26 | 🏆 **48.61** | high |
| Qwen | 🏆 **37.50** | 32.14 | 31.94 | low |
| DeepSeek-R1-671B | 🏆 **27.98** | 23.02 | 19.25 | low |

### 1.2.2 27-cell 细分与共性问题标记

共性标记方法：

- 对每个模型，取其 27 cells 中准确率落在该模型底部 25%（Q1）的 cells 记为 hard。  
- 对每个 cell 统计 hard 投票数（0 到 6）。  
- 当投票数 `>= 4/6` 时，记为共性问题 cell（✅）。

| cell_id | hard votes (out of 6) | common issue | mean_acc (%) | spread (pt) |
|---|---:|---|---:|---:|
| `alpha_high__disp_low__skew_low` | 🔴 **6** | ✅ | 25.30 | 17.86 |
| `alpha_high__disp_high__skew_mid` | 🔴 **5** | ✅ | 31.55 | 30.36 |
| `alpha_mid__disp_mid__skew_low` | 🔴 **5** | ✅ | 30.95 | 26.79 |
| `alpha_high__disp_high__skew_low` | 4 | ✅ | 32.44 | 21.43 |
| `alpha_mid__disp_high__skew_low` | 4 | ✅ | 28.87 | 10.71 |
| `alpha_mid__disp_low__skew_low` | 4 | ✅ | 31.25 | 17.86 |

按模型最难 cell（每模型 Top-1）：

| Model | hardest cell | acc (%) |
|---|---|---:|
| gpt4 | `alpha_high__disp_low__skew_low` | 26.79 |
| ollama-qwen2-7b | `alpha_high__disp_low__skew_low` | 30.36 |
| ollama-llama3-1-latest | `alpha_mid__disp_low__skew_low` | 25.00 |
| ollama-llama3-8b | `alpha_high__disp_low__skew_low` | 19.64 |
| Qwen | `alpha_low__disp_low__skew_high` | 17.86 |
| DeepSeek-R1-671B | `alpha_high__disp_low__skew_mid` | 10.71 |

可见：

- 大多数模型的最难点落在 high/mid `alpha` 且 `skew_low` 一带。  
- `alpha_high__disp_low__skew_low` 是最稳定的共性难点（6/6 投票）。


对后续微调的直接启发：

- 先用跨模型共同困难的 low-skew cells 构造 shared hard set。  
- 再用每模型 `delta_vs_cell_mean` 最弱的 cells 做 model-specific hard mining。  
- 训练目标优先抑制 `A_peer_weighted` 被 `top/closest`、`uniform avg`、`private baseline` 替代的现象。

## 4) 主要错误模式（细粒度表格）

来源：`analysis/model_performance_error_analysis/error_cause_summary.csv`

### 4.1 Eval-A 细粒度错因分布（占该模型 Eval-A 错题比例，%）

| Model | Top/closest | Copy closest | Uniform avg | Median | Private baseline | Equal mix (A->H) | Counter (A->G) | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt4 | 🔴 **27.60** | 🟢 **20.90** | 3.55 | 6.69 | 18.44 | 🔵 **22.27** | 0.55 | 0.00 |
| ollama-qwen2-7b | 🟢 **16.23** | 🔵 **17.72** | 9.54 | 15.24 | 🔴 **25.65** | 14.00 | 1.61 | 0.00 |
| ollama-llama3-1-latest | 🔵 **17.37** | 🟢 **16.00** | 7.60 | 11.24 | 🔴 **27.13** | 12.83 | 7.83 | 0.00 |
| ollama-llama3-8b | 🟢 **15.08** | 🔵 **17.41** | 6.43 | 13.08 | 🔴 **27.94** | 13.97 | 6.10 | 0.00 |
| Qwen | 🔵 **17.60** | 14.60 | 🔴 **20.50** | 🟢 **17.40** | 10.30 | 15.50 | 3.40 | 0.70 |
| DeepSeek-R1-671B | 15.89 | 10.79 | 🔴 **21.16** | 🔵 **18.39** | 8.20 | 🟢 **17.10** | 8.29 | 0.17 |

图例：🔴 = Top1，🔵 = Top2，🟢 = Top3。

### 4.2 其它任务的平均错因占比（跨模型均值，%）

> 下面用于观察任务层面最常见失败机制，不是单模型画像。

| Split | Error cause | Mean share (%) |
|---|---|---:|
| eval_B | misreads closeness-weight redistribution | 25.48 |
| eval_B | misses private-baseline F change | 23.53 |
| eval_B | misses social-sensitivity alpha change | 23.36 |
| eval_B | format / parse failure | 16.12 |
| eval_B | misses peer-action level change | 11.51 |
| eval_B | misses weighted reference aggregate change | 10.75 |
| placebo_test | uses social shortcut although gold is private baseline | 53.36 |
| placebo_test | hallucinates peer-weighted pull in placebo | 26.90 |
| placebo_test | other placebo confusion: D_pure_private -> H_equal_mix | 13.99 |
| placebo_test | other placebo confusion: D_pure_private -> G_counter_conformist | 6.15 |
| placebo_test | format / parse failure | 1.87 |
| ood_social | misses b/c matching stability criterion | 99.85 |
| ood_social | format / parse failure | 0.46 |
| ood_career | overweights relative status versus Langtry threshold | 48.22 |
| ood_career | overweights absolute salary / prestige | 47.37 |
| ood_career | career threshold tradeoff error | 5.08 |
| ood_career | format / parse failure | 3.07 |

总结：错误并非随机噪声，而是稳定的规则替代。`eval_A` 里主要是把 `A_peer_weighted` 替换成 `top/closest anchor`、`uniform average`、`private baseline` 或 `equal mix`；`placebo_test` 里则明显存在过度社会化，即该私有时仍套社会比较。

## 5) 可复现输入与导出文件

- 原始预测：`evaluation/outputs/<model>/*_predictions.jsonl`
- 分析 notebook：`notebooks/model_performance_error_analysis.ipynb`
- 导出目录：`analysis/model_performance_error_analysis/`
  - `accuracy_by_model_task.csv`
  - `model_gap_by_task.csv`
  - `error_cause_summary.csv`
  - `rule_confusions.csv`
  - `cross_model_item_disagreement.csv`
  - `response_marker_summary.csv`

---

## 总结

本评测在统一样本规模下（`eval_A=1512`，`eval_B=280 pairs`，`placebo=504`，`ood_social=576`，`ood_career=576`），系统检验了模型对 Langtry 结构决策函数的恢复能力。核心结果是，规则识别仍是主要瓶颈：`eval_A` 最优仅为 `51.59`（gpt4），最差为 `23.41`，任务内差距 `28.17` 个百分点，说明多数模型尚未稳定掌握 closeness 加权聚合机制。

比较静态任务相对更容易：`eval_B` 最优 `70.00`（Qwen），最差 `56.43`，差距 `13.57` 个百分点，表明模型可部分把握方向变化，但这不足以证明其已学会完整决策函数。安慰剂任务显示显著门控差异：`placebo_test` 最优 `96.03`（gpt4），最差 `29.96`，差距 `66.07` 个百分点，提示不同模型在抑制过度社会化方面能力悬殊。

OOD 结果进一步表明，分布外挑战不仅来自场景语义变化，还来自决策函数形式变化。`ood_social` 最优 `69.44`、最差 `56.94`，仍有稳定误差；`ood_career` 最优 `51.74`、最差 `45.83`，整体接近机会水平，说明在绝对收益与相对地位冲突下的阈值权衡仍不稳健。

分层分析支持上述判断：在 Eval-A 的 27-cell 网格中，共性难点集中在高或中 `alpha` 且 `skew_low` 的区域，其中 `alpha_high__disp_low__skew_low` 被 `6/6` 模型投票为共性 hard cell。结合错因分布，失败并非噪声，而是可重复的规则替代现象，例如将 gold 规则替换为 top-anchor、uniform-average 或 private-baseline。该结构化误差谱可直接转化为后续微调中的 hard mining 与对比偏好构造。
