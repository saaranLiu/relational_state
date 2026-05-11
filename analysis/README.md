# Relational-State 评测结果

本文件汇总当前仓库中最新评测输出（`evaluation/outputs/*`）对应的主要结果，用于论文写作与后续微调设计参考。


## 理论基础公式（Langtry）

主任务的核心决策函数为：

`x_i^* = F_i + alpha_i * R_i`，其中社会参照 `R_i = Σ_j g_ij * x_j`（对同伴行动 `x_j` 按亲疏权重 `g_ij` 加权求和）。

等价展开：`x_i^* = F_i + alpha_i * Σ_j g_ij * x_j`。

含义：个体先有私有基线 `F_i`，再按 `g_ij` 聚合同伴行动得到 `R_i`，并由社会敏感度 `alpha_i` 调节社会项强度。

## 1) 评测任务说明

- `eval_A`: 规则识别（是否选择 theory-based 规则）
- `eval_B`: 比较静态方向判断（pairwise, 判断单调性）
- `placebo_test`: 非社会比较场景（ground truth 为私有基线）
- `ood_social`: OOD 社交匹配任务（基于 `b/c` 匹配稳定性）
- `ood_career`: OOD 职业排序任务（绝对薪资 vs 相对地位阈值权衡）

### 1.0) 各任务样本数

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

规则池（A-H）语义如下。说明：每题为「1 个 gold 规则 + 3 个 distractor」自池中按均匀与可辨识性约束采样；**同一叙事场景（`scene_id`）**可在多题中复用，但各题的选项字母、规则组合与 27-cell 潜变量配置仍由生成与均衡脚本独立控制，二者并非简单一一对应。

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

### E) 采样规则

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
- 对微调价值：可作为多目标冲突下的压力测试，便于构造「绝对收益 vs 相对地位」类对比偏好数据。

### F) 组合解释

- `eval_A` 高 + `placebo_test` 高：说明模型既会用规则，也会在不该用时停用（有门控）。我们预期能够看到的微调效果是各任务提升的同时，placebo test尽量不降低。
- `eval_B` 高但 `eval_A` 低：说明仅有方向性启发式，不具备完整机制识别。
- `ood_social` / `ood_career` 等综合 OOD 任务表现高：说明结构能力可迁移；表现低：说明仍偏模板依赖或对新决策函数形式不稳健。

因此，本评测体系的价值在于把是否具备社会比较推理能力拆解为：  
**规则识别、方向敏感、门控抑制、跨任务迁移、冲突权衡** 五个维度，并将错误直接转化为后续微调目标。

## 2) 各模型主结果（Accuracy, %）

来源：`analysis/model_performance_error_analysis/accuracy_by_model_task.csv`

| Model | eval_A | eval_B | placebo_test | ood_social | ood_career |
|---|---:|---:|---:|---:|---:|
| ollama-qwen2.5-14b | 🏆 **60.91** | 66.07 | 39.48 | 66.49 | 49.65 |
| gpt4 | 51.59 | 62.50 | 🏆 **96.03** | 69.44 | 50.00 |
| Qwen | 49.80 | 65.36 | 67.26 | 65.45 | 50.35 |
| ollama-qwen2-7b | 46.63 | 56.43 | 91.67 | 59.20 | 47.57 |
| DeepSeek-V4-Flash | 45.97 | 🏆 **71.79** | 64.48 | 🏆 **74.83** | 49.31 |
| ollama-llama3-1-latest | 41.73 | 61.79 | 75.20 | 61.98 | 50.35 |
| ollama-llama3-8b | 40.34 | 58.21 | 79.37 | 59.72 | 🏆 **51.74** |

上表含 `ollama-qwen2.5-14b`（来自 `evaluation/outputs/ollama-qwen2.5-14b/*_summary.json` 与 `eval_*_predictions.jsonl`）；`accuracy_by_model_task.csv` 若未列出该行，可运行 `scripts/refresh_model_performance_csvs.py` 刷新。

图例：🏆 表示该任务下最高准确率。

### 2.1) 跨模型共同错题与推理（Reasoning）侧写

**定义（共同错题 / 「全错」）**：对同一 `task_id`，若当前仓库中 **全部已纳入统计的模型目录**（`scripts/analyze_common_wrong_cross_model.py` 扫描 `evaluation/outputs/*/*_predictions.jsonl`，排除 `gpt-5.5/`、`temp/`）下该 split 的预测均为错误（`is_correct` 为 `False` 或解析失败视为错），则记为该 split 的一条「全错」样本。当前 **`eval_A` 上为 7 路模型**：`DeepSeek-V4-Flash`、`Qwen`、`gpt4`、`ollama-llama3-1-latest`、`ollama-llama3-8b`、`ollama-qwen2-7b`、`ollama-qwen2.5-14b`。下文为叙述方便仍可能写作「全错子集」；与旧版「六模型」口径相比，仅多一路本地 Qwen2.5-14B。

**各 split 全错占比**（可复现脚本：`scripts/analyze_common_wrong_cross_model.py`；表头沿用 CSV 列名 `universal_wrong_count`，含义为「当前模型集合下一路不漏全错」）：

| dataset_split | n_tasks | 全错条数 | 比例 |
|---|---:|---:|---:|
| eval_A | 1,512 | 126 | 8.33% |
| eval_B | 280 | 8 | 2.86% |
| placebo_test | 504 | 2 | 0.40% |
| ood_social | 576 | 49 | 8.51% |
| ood_career | 576 | 237 | 41.15% |

导出：`analysis/model_performance_error_analysis/common_wrong_by_split.csv`；`eval_A` 上每条全错任务的逐题明细（含各模型截断推理片段）见 `common_wrong_eval_A_universal.csv`。

**推理文本来源**：评测写入的 `raw_response`（`eval_runner.py` 将模型完整输出存于此字段；部分 API 的 `reasoning_content` 若被合并进最终 `content`，也会出现在同一字符串中）。脚本优先截取 `Reasoning:` 至 `Choice:` 之间的段落；若无该标记则取全文前缀用于关键词统计（长链式推理模型适用）。

**在 126 条 `eval_A` 全错题目上，错选规则（`parsed_rule_id`）在直方图汇总的 878 次模型回答中的占比**（由各题 `parsed_rule_histogram` 累加，可解析项不足 7 的题会使总数略小于 126×7）：`E_closest_mimicry` 约 **25.7%**；`H_equal_mix` 约 **22.1%**；`B_top_anchor` 约 **21.4%**；`D_pure_private` 约 **16.2%**；其余为 `F_median_anchor`（约 9.0%）、`C_uniform_avg`、`G_counter_conformist` 等。说明共同失败仍是 **gold `A_peer_weighted`（亲疏加权聚合同伴）被单锚、折中、私域或匀权等捷径规则系统性替换**，而非随机字母噪声。

**对推理文本的轻量关键词覆盖**（每条模型回答可对多个关键词各计 1 次；原始计数见 `common_wrong_eval_A_reasoning_keywords.csv`，随全错子集与模型数重算）：

| 关键词主题 | 命中次数（跨 126×7 条推理，当前 CSV） |
|---|---:|
| 社会参照 / 同伴压力叙事（peer、crowd、visible、norm 等） | 823 |
| 私人基线 / 独处叙事（private、baseline、alone 等） | 691 |
| 权重 / 亲密度 / 关系强度措辞（weight、closeness、bond 等） | 526 |
| 匹配 / 模仿 / 对齐叙事（match、mirror、copy 等） | 415 |
| 最近 / 最亲 / 单一锚点叙事（closest、mentor、roommate 等） | 409 |
| 平均 / 典型 / 中位叙事（average、mean、median、typical 等） | 134 |

**解读（与错因表对照）**：叙事里大量出现「同伴可见度、社会规范」与「私人基线」两类线索，与模型在规则层把 `A_peer_weighted` 错成 `B_top_anchor`、`E_closest_mimicry`、`D_pure_private`、`H_equal_mix`、`C_uniform_avg` 等现象一致——即推理往往在讲 **社会比较故事或关系亲疏故事**，但最终选到的 **离散规则标签** 仍落在更粗的捷径规则上，而不是正确的 closeness 加权聚合。该侧写可用于微调：对全错子集做 hard negative（显式对比「加权聚合 vs 单锚 / 均匀 / 纯私域」的 CoT 偏好）。

#### 2.1.1 Reasoning 深度特征（结构/语言线索，可复现）

脚本：`scripts/analyze_reasoning_deep_eval_a.py`。在每条 `eval_A` 预测的 `Reasoning:` 文本上计算：长度与句数；若干 **布尔线索**（是否出现「单最近同伴 / 模仿」类措辞、`each parent / weighted aggregate` 等多主体加权类措辞、是否出现美元与算术推敲、「折中/ torn between」类措辞、是否出现「坚持私域/无视同伴」类措辞等）；以及 **尾部自述选项字母** 与 `Choice:` 行字母是否一致（启发式，对极长链式模型覆盖率低，见 `reasoning_deep_eval_a_tail_letter_mismatch.csv`）。

**各模型在「答错」子集上的线索命中率（Eval-A 全量，1512 题/模型）** 摘要：

- **DeepSeek-V4-Flash / Qwen**：错题推理仍 **偏长**（错题平均字符约 **2,511 / 2,248**），且 **数值推敲**（`p_numeric_when_wrong`）分别约 **0.66 / 0.59**；DeepSeek 错题里 **多主体加权话术**（`p_multi_when_wrong`）约 **0.52**，Qwen 约 **0.31**。在 **输出长度受限、解析基本成功** 的设定下，失败更多体现为 **长叙事仍不能稳定收束到 gold 规则**（映射/判别错误），而非单纯的截断丢 `Choice:`。
- **gpt-4**：错题推理 **短**（约 **318** 字符量级），**单锚/模仿** 与 **多主体加权** 两类触发率都 **低**（错题里约 **0.22 / 0.02**），更像 **直接跳到结论式捷径**，而非展开演算。
- **三台 Ollama 小参**：错题推理 **短**（约 **200–300** 字符），**多主体加权话术几乎不出现**（`p_multi_when_wrong` 约 **0.004–0.014**），**单锚类** 亦较低；**「折中」叙事** 在 llama3-8b 错题里相对突出（约 **0.33**）。整体更接近 **浅层模板句**，与 API 模型的错误 **形态不同**。

**在 126 道七路全错题上（大三档每档 378 条 = 126×3）**：`large_api` 三模型错题推理平均长度仍多在 **千字符量级**（随脚本 `analyze_reasoning_deep_eval_a.py` 版本与截断策略略变），`small_ollama`（三台最小本地）约 **数百字符**；定性结论与旧版一致：**最难题上大模型往往更长、更「机制化」却仍集体错；小模型更接近短模板**——二者都应针对 **规则绑定** 训练，但数据形态应 **分档设计**（长错链 vs 短模板）。若需与当前 `evaluation/outputs` 完全一致的数值，请重跑 `scripts/analyze_reasoning_deep_eval_a.py` 后读 `reasoning_deep_eval_a_small_vs_large_six_wrong.csv`（文件名仍含 `six`，对应「全错题」子集）。

**按错选规则条件概率（Eval-A 全部错题汇总，`reasoning_deep_eval_a_flags_by_top_wrong_rule.csv`）**：错成 **`H_equal_mix`** 时 **「折中/ torn」** 类措辞约 **51%**；错成 **`C_uniform_avg`** 时 **数值推敲** 约 **66%**；错成 **`E_closest_mimicry` / `B_top_anchor`** 时 **单最近同伴/模仿** 类措辞约 **37%–40%**。可将这些模式当作 **自动分层标签**，用于构造对比式训练样本（例如：同一题干下强制区分「算术平均」与「亲疏加权平均」的收尾句）。

#### 2.1.2 全错 Top5 定性材料（`evalA_reasoning.md`）

在 `eval_A` **当前模型集合下一路不漏全错**的子集中，按 `task_id` 升序 **贪心** 选取 **5** 题，使各题错选众数的 `parsed_rule_id` **尽量互不重复**；材料汇总为唯一维护文件 **`analysis/evalA_reasoning.md`**：含 **数据与选题说明、题目表、每题 gold + 字母→规则表、完整题干（内嵌 `gpt-4` 行中的 `prompt`，与评测一致）、各模型 `raw_response` 推理片段与 Choice**；推理正文为 **`Reasoning:` 至其后首个 `Choice:`** 或 **全文至最后一个 `Choice:` 之前**（导出脚本不截断）。若 `raw_response` 缺 `Choice:`，块内 **`# --- 评测补写`** 以解析字母补全。

**机器稿重新生成**（**会整文件覆盖** `evalA_reasoning.md`，请先 git 备份）：`python3 scripts/export_top5_universal_evalA_reasoning.py`。仓库中当前版本在机器稿之外另含 **人工撰写的各段「简析」及本地小参数四模型总结**；若重跑脚本后需恢复，请从版本库检出或手工合并。

### 2.2) Llama / Qwen 三小参（Ollama）共同错题，及与「大三档」对比

**小参三模型**（本仓库中的本地 Llama 与 Qwen 小权重运行，与六模型全集对照）：`ollama-llama3-1-latest`、`ollama-llama3-8b`、`ollama-qwen2-7b`。  
**大三档三模型**（API / 大权重一档，脚本中仅用于对照统计）：`gpt4`、`Qwen`、`DeepSeek-V4-Flash`。二者各含 3 个检查点，便于在 **同一题集** 上做并列对照（注意：大三档内部能力差异仍明显，**三模型算术均值**不能等同于「大模型」单一概念，见下表后说明）。

**各 split「三小参全错」题数与比例**（定义与 §2.1 相同，只是把「6 个全错」换成「上述 3 个 Ollama 全错」）：

| dataset_split | n_tasks | 三小参全错条数 | 比例 |
|---|---:|---:|---:|
| eval_A | 1,512 | 423 | 27.98% |
| eval_B | 280 | 29 | 10.36% |
| placebo_test | 504 | 15 | 2.98% |
| ood_social | 576 | 115 | 19.97% |
| ood_career | 576 | 245 | 42.53% |

导出：`common_wrong_small3_by_split.csv`。

**与「七路全错」的包含关系（`eval_A`）**：`126` 道七路全错题 **全部** 落在「**三小参**（`ollama-llama3-1-latest`、`ollama-llama3-8b`、`ollama-qwen2-7b`）全错」集合之内；另有 **`239`** 道题（`1512` 中的 **15.81%**）满足 **三小参全错但至少有一路当前统计内的模型答对**（例如新纳入的 **`ollama-qwen2.5-14b`** 或其它大路模型捞回）。换言之，**最难子集仍嵌在三小参全挂集合中，但全错集合会因新增模型而收窄**。量化上，在大三档的 **逐条预测** 上，落在「三小参全错」题目集合时的平均正确率约为 **30.3%**（见 `common_wrong_eval_A_small3_overlap.csv` 中 `large_tier_mean_accuracy_on_eval_A_tasks_where_small3_all_wrong`）；而在 **七路全错** 题目上，所有已纳入模型的平均正确率必为 **0**（定义使然）。说明：重叠 CSV 中部分 metric 名仍含 `six_all_wrong`，与脚本历史命名一致，**数值行对应当前 7 路全错子集**。

**三小参全错时的错选规则倾向（仅 `eval_A`，共 1,269 次错误回答 = 423×3）**：错选分布中 **`D_pure_private` 约占 31.4%**，高于七路全错汇总里该规则约 **16.2%** 的占比，说明 **小参本地模型更常把题解成「纯私域基线」**；其次为 **`E_closest_mimicry`、`B_top_anchor`、`H_equal_mix`** 等锚点 / 均匀类捷径（详见 `common_wrong_eval_A_small_vs_large_rules.csv` 中 `three_small_on_small3_wrong_tasks`）。

**同一批「三小参全错」题上，大三档仅在错误回答内的规则分布**（剔除答对的 `A_peer_weighted`，共 923 次大三档错题）：首位为 **`H_equal_mix`（约 24.6%）**，其次 **`E_closest_mimicry`（约 24.1%）**、`B_top_anchor`（约 22.7%）、`D_pure_private`（约 14.8%）等。在 **七路全错** 子集上，大三档错题分布与七路汇总一致（因大三档亦全错），见同一 CSV 中 `three_large_wrong_only_on_six_wrong_tasks` 与 `three_small_on_six_wrong_tasks` 两行块对照：小参侧 **`D_pure_private` 仍更重**（约 **28.9%** 对 **14.5%**），大参侧错题更向 **`E_closest_mimicry`、`H_equal_mix`、`B_top_anchor`** 倾斜。

**小参 vs 大参「还有什么区别」**（结合 `small_vs_large_mean_accuracy_by_split.csv` 的三模型均值）：

| dataset_split | 三小参均值准确率 | 大三档均值准确率 |
|---|---:|---:|
| eval_A | 42.90% | 49.12% |
| eval_B | 58.81% | 66.55% |
| placebo_test | 82.08% | 75.93% |
| ood_social | 60.30% | 69.91% |
| ood_career | 49.88% | 49.88% |

解读要点：**在将 API 端点从 `DeepSeek-R1-671B` 更新为 `DeepSeek-V4-Flash` 并收紧输出后，`eval_A` 上大三档均值已高于三小参**；此前「大三档均值被 DeepSeek-R1 在规则识别任务上的极低分拖累」的现象主要来自旧端点与长输出截断。**`placebo_test`** 上三小参均值仍高于大三档，提示 **门控 / 私域基线** 仍是云端组合里相对更难的一块，但差距已明显缩小。错选谱上，**小参仍更偏向 `D_pure_private`，大参在仍会错时更偏向 `E_closest_mimicry` / `H_equal_mix` / `B_top_anchor`**，可作为分档微调与蒸馏时不同的负样本权重参考。

## 3) 任务级 best/worst 与区分度（spread）

来源：`analysis/model_performance_error_analysis/model_gap_by_task.csv`

- `eval_A`: best = **51.59** (`gpt4`), worst = 40.34, spread = **11.24**
- `eval_B`: best = **71.79** (`DeepSeek-V4-Flash`), worst = 56.43, spread = **15.36**
- `placebo_test`: best = **96.03** (`gpt4`), worst = 64.48, spread = **31.55**
- `ood_social`: best = **74.83** (`DeepSeek-V4-Flash`), worst = 59.20, spread = **15.63**
- `ood_career`: best = **51.74** (`ollama-llama3-8b`), worst = 47.57, spread = **4.17**

结论：`eval_A` 仍是核心结构能力的区分任务，但在六模型可比集合上 **头尾差距收窄**（最差模型已回到约 40% 而非 ~23%）；`placebo_test` 在是否抑制不该有的社会比较上仍拉开较大差距；`ood_career` 全体仍接近 chance 区间。

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

### 3.1.1 分模型参数统计（不用总体平均）

#### A) `alpha` 分桶准确率（%）

| Model | low | mid | high | best bucket |
|---|---:|---:|---:|---|
| gpt4 | 🏆 **63.89** | 49.40 | 41.47 | low |
| ollama-qwen2-7b | 🏆 **53.97** | 43.65 | 42.26 | low |
| ollama-llama3-1-latest | 🏆 **45.44** | 40.67 | 39.09 | low |
| ollama-llama3-8b | 🏆 **45.83** | 37.30 | 37.90 | low |
| Qwen | 🏆 **60.12** | 48.81 | 40.48 | low |
| DeepSeek-V4-Flash | 🏆 **62.10** | 39.68 | 36.11 | low |

#### B) `dispersion` 分桶准确率（%）

| Model | low | mid | high | best bucket |
|---|---:|---:|---:|---|
| gpt4 | 🏆 **52.38** | 50.40 | 51.98 | low |
| ollama-qwen2-7b | 46.63 | 44.84 | 🏆 **48.41** | high |
| ollama-llama3-1-latest | 40.87 | 42.06 | 🏆 **42.26** | high |
| ollama-llama3-8b | 38.49 | 41.07 | 🏆 **41.47** | high |
| Qwen | 33.73 | 🏆 **37.90** | 29.96 | mid |
| DeepSeek-V4-Flash | 45.44 | 45.04 | 🏆 **47.42** | high |

#### C) `skew` 分桶准确率（%）

| Model | low | mid | high | best bucket |
|---|---:|---:|---:|---|
| gpt4 | 44.25 | 53.97 | 🏆 **56.55** | high |
| ollama-qwen2-7b | 41.07 | 46.83 | 🏆 **51.98** | high |
| ollama-llama3-1-latest | 36.90 | 42.26 | 🏆 **46.03** | high |
| ollama-llama3-8b | 30.16 | 42.26 | 🏆 **48.61** | high |
| Qwen | 47.22 | 50.60 | 🏆 **51.59** | high |
| DeepSeek-V4-Flash | 41.27 | 47.62 | 🏆 **49.01** | high |

### 3.1.2 27-cell 细分与共性问题标记

共性标记方法：

- 对每个模型，取其 27 cells 中准确率落在该模型底部 25%（Q1）的 cells 记为 hard。  
- 对每个 cell 统计 hard 投票数（0 到 6）。  
- 当投票数 `>= 4/6` 时，记为共性问题 cell（✅）。

下表为 **全部 27 个 cell**（按 hard 投票数降序、再按跨模型平均准确率升序）。`mean_acc` 与 `spread` 来自 `evalA_cell_commonality.csv`（六模型在各 cell 上准确率的均值与 max−min，单位均为百分点）；`hard votes` 为「该 cell 落在该模型 27-cell 准确率下四分位（最难 25%）」的模型计数（0–6）；`common issue` 表示 `hard votes ≥ 4` 的共性难点 cell。

| cell_id | hard votes (out of 6) | common issue | mean_acc (%) | spread (pt) |
|---|---:|---|---:|---:|
| `alpha_mid__disp_low__skew_low` | 🔴 **6** | ✅ | 26.79 | 23.21 |
| `alpha_high__disp_low__skew_low` | 🔴 **6** | ✅ | 28.57 | 21.43 |
| `alpha_mid__disp_high__skew_low` | 🔴 **6** | ✅ | 31.55 | 16.07 |
| `alpha_high__disp_high__skew_mid` | 5 | ✅ | 33.04 | 26.79 |
| `alpha_high__disp_mid__skew_low` | 5 | ✅ | 33.33 | 10.71 |
| `alpha_high__disp_high__skew_low` | 5 | ✅ | 37.50 | 14.29 |
| `alpha_mid__disp_mid__skew_low` | 4 | ✅ | 36.01 | 17.86 |
| `alpha_high__disp_mid__skew_mid` | 1 | — | 41.07 | 12.50 |
| `alpha_high__disp_low__skew_mid` | 1 | — | 41.37 | 12.50 |
| `alpha_mid__disp_high__skew_mid` | 1 | — | 45.24 | 25.00 |
| `alpha_high__disp_high__skew_high` | 1 | — | 45.83 | 21.43 |
| `alpha_mid__disp_mid__skew_mid` | 1 | — | 46.73 | 23.21 |
| `alpha_low__disp_high__skew_low` | 1 | — | 52.08 | 42.86 |
| `alpha_low__disp_high__skew_high` | 1 | — | 54.46 | 28.57 |
| `alpha_mid__disp_mid__skew_high` | 0 | — | 44.05 | 3.57 |
| `alpha_high__disp_mid__skew_high` | 0 | — | 46.13 | 12.50 |
| `alpha_high__disp_low__skew_high` | 0 | — | 49.11 | 17.86 |
| `alpha_low__disp_mid__skew_mid` | 0 | — | 50.60 | 7.14 |
| `alpha_mid__disp_low__skew_high` | 0 | — | 51.49 | 21.43 |
| `alpha_mid__disp_low__skew_mid` | 0 | — | 51.49 | 21.43 |
| `alpha_low__disp_low__skew_high` | 0 | — | 52.08 | 19.64 |
| `alpha_low__disp_low__skew_mid` | 0 | — | 52.38 | 28.57 |
| `alpha_low__disp_mid__skew_low` | 0 | — | 54.46 | 35.71 |
| `alpha_mid__disp_high__skew_high` | 0 | — | 55.95 | 16.07 |
| `alpha_low__disp_mid__skew_high` | 0 | — | 56.55 | 19.64 |
| `alpha_low__disp_low__skew_low` | 0 | — | 61.01 | 32.14 |
| `alpha_low__disp_high__skew_mid` | 0 | — | 63.39 | 25.00 |

逐模型、逐 cell 的样本量与准确率见 **`evalA_by_27_cells.csv`**。

按模型最难 cell（每模型 Top-1）：

| Model | hardest cell | acc (%) |
|---|---|---:|
| gpt4 | `alpha_high__disp_low__skew_low` | 26.79 |
| ollama-qwen2-7b | `alpha_high__disp_low__skew_low` | 30.36 |
| ollama-llama3-1-latest | `alpha_mid__disp_low__skew_low` | 25.00 |
| ollama-llama3-8b | `alpha_high__disp_low__skew_low` | 19.64 |
| Qwen | `alpha_low__disp_low__skew_high` | 17.86 |
| DeepSeek-V4-Flash | `alpha_mid__disp_low__skew_low` | 12.50 |

可见：

- 大多数模型的最难点仍集中在 **低 skew** 或 **中高 alpha** 相关 cells。  
- `alpha_mid__disp_low__skew_low`、`alpha_high__disp_low__skew_low`、`alpha_mid__disp_high__skew_low` 等为 **6/6 hard 投票** 的共性难点（见上表）。


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
| Qwen | 🔴 **26.75** | 🔵 **23.45** | 9.75 | 11.86 | 7.91 | 15.42 | 1.71 | 3.16 |
| DeepSeek-V4-Flash | 🔵 **22.28** | 20.44 | 9.55 | 10.89 | 4.65 | 🔴 **25.58** | 4.04 | 2.57 |

图例：🔴 = Top1，🔵 = Top2，🟢 = Top3。

### 4.2 其它任务的平均错因占比（跨模型均值，%）

> 下面用于观察任务层面最常见失败机制，不是单模型画像。

| Split | Error cause | Mean share (%) |
|---|---|---:|
| eval_B | misses social-sensitivity alpha change | 26.78 |
| eval_B | misreads closeness-weight redistribution | 25.10 |
| eval_B | misses private-baseline F change | 22.77 |
| eval_B | format / parse failure | 16.50 |
| eval_B | misses peer-action level change | 10.88 |
| eval_B | misses weighted reference aggregate change | 8.97 |
| placebo_test | uses social shortcut although gold is private baseline | 50.18 |
| placebo_test | hallucinates peer-weighted pull in placebo | 31.38 |
| placebo_test | other placebo confusion: D_pure_private -> H_equal_mix | 13.36 |
| placebo_test | other placebo confusion: D_pure_private -> G_counter_conformist | 5.07 |
| placebo_test | format / parse failure | 2.59 |
| ood_social | misses b/c matching stability criterion | 97.90 |
| ood_social | format / parse failure | 6.31 |
| ood_career | overweights relative status versus Langtry threshold | 97.67 |
| ood_career | format / parse failure | 1.56 |
| ood_career | overweights absolute salary / prestige | 1.40 |
| ood_career | career threshold tradeoff error | 1.31 |

总结：错误并非随机噪声，而是稳定的规则替代。`eval_A` 里主要是把 `A_peer_weighted` 替换成 `top/closest anchor`、`uniform average`、`private baseline` 或 `equal mix`；`placebo_test` 里则明显存在过度社会化，即该私有时仍套社会比较。

## 5) 可复现输入与导出文件

- 原始预测：`evaluation/outputs/<model>/*_predictions.jsonl`
- 分析 notebook：`notebooks/model_performance_error_analysis.ipynb`
- 导出目录：`analysis/model_performance_error_analysis/`
  - `accuracy_by_model_task.csv`、`model_gap_by_task.csv`、`error_cause_summary.csv`、`evalA_by_27_cells.csv`、`evalA_cell_commonality.csv`、`evalA_cell_readme_order.csv`（由 `scripts/refresh_model_performance_csvs.py` 从当前 `evaluation/outputs` 重算，默认跳过 `gpt-5.5/`、`temp/`）
  - `rule_confusions.csv`、`cross_model_item_disagreement.csv`、`response_marker_summary.csv`（仍为 notebook 侧导出；若需与当前 `evaluation/outputs` 完全一致请本地重跑 `notebooks/model_performance_error_analysis.ipynb`）
  - `common_wrong_by_split.csv`、`common_wrong_eval_A_universal.csv`、`common_wrong_eval_A_reasoning_keywords.csv`（当前模型集合下「全错」+ 推理关键词）
  - `common_wrong_small3_by_split.csv`、`common_wrong_eval_A_small3_overlap.csv`、`common_wrong_eval_A_small_vs_large_rules.csv`、`small_vs_large_accuracy_by_model_split.csv`、`small_vs_large_mean_accuracy_by_split.csv`（三小参 Ollama 全错 + 与大三档对照；同上脚本生成）
  - `reasoning_deep_eval_a_per_row.csv`、`reasoning_deep_eval_a_by_model.csv`、`reasoning_deep_eval_a_by_parsed_rule_wrong.csv`、`reasoning_deep_eval_a_small_vs_large_six_wrong.csv`、`reasoning_deep_eval_a_flags_by_top_wrong_rule.csv`、`reasoning_deep_eval_a_tail_letter_mismatch.csv`（Eval-A Reasoning 深度特征；`scripts/analyze_reasoning_deep_eval_a.py`）
  - `analysis/evalA_reasoning.md`（`eval_A` 全错 Top5 定性材料；机器稿由 `scripts/export_top5_universal_evalA_reasoning.py` 生成，仓库版本常含人工简析）

---

## 总结

本评测在统一样本规模下（`eval_A=1512`，`eval_B=280 pairs`，`placebo=504`，`ood_social=576`，`ood_career=576`），系统检验了模型对 Langtry 结构决策函数的恢复能力。核心结果是，规则识别仍是主要瓶颈：在纳入 **`ollama-qwen2.5-14b`** 的当前 README 主表上，`eval_A` 最优约 **`60.91`**（本地 14B），`gpt4` 约 **`51.59`**，最差约 **`40.34`**（`ollama-llama3-8b`），任务内差距约 **20.6** 个百分点（排除实验目录 `gpt-5.5/`、`temp/`）；说明 **多数检查点仍未稳定掌握 closeness 加权聚合**，且本地 14B 在 `eval_A` 上高于 API `gpt4` 的现象与 **`placebo_test` 上 14B 明显偏低（约 39%）** 并存，提示 **门控与规则识别可能不同步**，不宜以单 split 排序替代整体结论。

比较静态任务上，`eval_B` 最优约 `71.79`（`DeepSeek-V4-Flash`），最差 `56.43`，差距约 `15.36` 个百分点，表明模型可把握部分方向变化，但仍不足以证明已学会完整决策函数。安慰剂任务显示门控差异：`placebo_test` 最优 `96.03`（gpt4），最差约 `64.48`，差距约 `31.55` 个百分点。

OOD 结果进一步表明，分布外挑战不仅来自场景语义变化，还来自决策函数形式变化。`ood_social` 最优约 `74.83`、最差约 `59.20`；`ood_career` 最优 `51.74`、最差约 `47.57`，整体仍接近机会水平，说明在绝对收益与相对地位冲突下的阈值权衡仍不稳健。

分层分析支持上述判断：在 Eval-A 的 27-cell 网格中，共性难点仍集中在 **低 skew / 中高 alpha** 一带，且当前数据下 **`alpha_mid__disp_low__skew_low` 等 cells 取得 hard 投票满票**（见 `evalA_cell_readme_order.csv` 与上表；hard 投票计数随纳入 `evaluation/outputs` 的模型数变化）。结合错因分布，失败并非噪声，而是可重复的规则替代现象。该结构化误差谱可直接转化为后续微调中的 hard mining 与对比偏好构造。

在题目层面，约 **8.33%** 的 `eval_A` 题（126 / 1,512）出现 **当前 7 路模型全错**；对这些题的推理文本进行关键词侧写可见，叙述仍大量围绕同伴与亲疏，但离散选项却系统性落在模仿最近同伴、折中/均匀/中位锚、纯私域等捷径规则上，进一步支持「叙事社会性 ≠ 选对决策函数」这一微调切入点。
