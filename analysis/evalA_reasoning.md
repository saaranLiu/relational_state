# eval_A 七路模型全错 · Top 5 题

**数据**：`common_wrong_eval_A_universal.csv`（当前 **126** 道在 `eval_A` 上 **7** 路模型预测均错误的题目；在既有 6 路基础上纳入 **`ollama-qwen2.5-14b`** 后，「全错」集合从 166 收窄为 126）。

**选题**：按 `task_id` **升序**遍历，选取 **5** 题，使各题错误答案众数对应的 **`parsed_rule_id` 尽可能互不重复**。


## 题目

| # | `task_id` | `cell_id` | gold | 众数错选 | 错选（7 次） |
|---:|---|---|---|---|---|
| 1 | `evalA__d1_cele_02__alpha_high__disp_high__skew_low__00` | `alpha_high__disp_high__skew_low` | `A`→`A_peer_weighted` | `E_closest_mimicry` | `E_closest_mimicry`×4, `B_top_anchor`×2, `C_uniform_avg`×1 |
| 2 | `evalA__d1_cele_02__alpha_high__disp_high__skew_low__01` | `alpha_high__disp_high__skew_low` | `B`→`A_peer_weighted` | `F_median_anchor` | `F_median_anchor`×3, `D_pure_private`×2, `E_closest_mimicry`×2 |
| 3 | `evalA__d1_cele_02__alpha_high__disp_low__skew_low__00` | `alpha_high__disp_low__skew_low` | `A`→`A_peer_weighted` | `H_equal_mix` | `H_equal_mix`×5, `D_pure_private`×2 |
| 4 | `evalA__d1_cele_02__alpha_high__disp_mid__skew_mid__00` | `alpha_high__disp_mid__skew_mid` | `A`→`A_peer_weighted` | `B_top_anchor` | `B_top_anchor`×3, `F_median_anchor`×3, `C_uniform_avg`×1 |
| 5 | `evalA__d1_dining_03__alpha_high__disp_low__skew_high__01` | `alpha_high__disp_low__skew_high` | `B`→`A_peer_weighted` | `D_pure_private` | `D_pure_private`×6, `C_uniform_avg`×1 |

---

## 1. `evalA__d1_cele_02__alpha_high__disp_high__skew_low__00`

`scene_id`：**d1_cele_02** · `cell_id`：**alpha_high__disp_high__skew_low** · 错选直方图：`E_closest_mimicry`×4, `B_top_anchor`×2, `C_uniform_avg`×1 · 众数：**`E_closest_mimicry`**

**Gold**：`A` → `A_peer_weighted`。

| 字母 | `parsed_rule_id` |
|---|---|
| `A` | `A_peer_weighted` |
| `B` | `C_uniform_avg` |
| `C` | `B_top_anchor` |
| `D` | `E_closest_mimicry` |

**题干**（摘自 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中该 `task_id` 的 `prompt` 字段，与评测时模型所见一致）：

```text
Scenario:
Our young-parents group chat lit up planning first birthday parties, with past bashes memorialized on Instagram. I’d quietly budgeted $3,486—enough for a backyard bash with cupcakes and a balloon arch, no fuss. But Maya, my college roommate turned mom-confidante, dropped her plans: custom smash cakes, pony rides, and a photo booth tallying $4,924. Meanwhile, Ben (that quiet dad from the pediatrician’s Slack) mentioned his low-key zoo trip at $3,112. Chloe’s Instagram stories flaunted her $3,504 floral-themed brunch, and Dave—someone I only recognize from daycare pickup—chatted about his $3,561 carnival setup. Now the group’s buzzing with Pinterest links and venue debates. Maya’s extravagance glows in the chat, while others hover near my number. Should I upgrade the pony to match her or stick to my quiet plan?

Candidate rules for party budget:
(A) Build on the baseline with a social contribution in which close ties weigh heavier than background acquaintances — lands near $7,441.
(B) Take the unweighted mean of what the parents are doing and stack that social term above the private floor — arrives near $6,789.
(C) Route the peer component entirely through the most relationally close parent and layer it on top of the personal floor — around $7,793.
(D) Replace the private baseline with the single most intimate parent's move and stop there — ends at $4,924.

Output format (STRICT — do not add extra lines):
Reasoning: <exactly 1 to 3 sentences, no more>
Choice: <one letter: A, B, C, or D>
```


### 各模型输出

#### DeepSeek-V4-Flash

`parsed_letter` → `parsed_rule_id`：**D** → **E_closest_mimicry**

```text
Situation: The protagonist has a private budget of $3,486 but Maya, a close friend, plans a $4,924 party; others spend near the protagonist's baseline.
Psychology: The protagonist feels social pressure from Maya's extravagance and considers upgrading to match her, weighing close ties over acquaintances.
Choice_reason: The rule that replaces the private baseline with the most intimate parent's move best captures matching Maya's plan.

Choice: D
```

**简析**：把社会项写成「用关系最亲的家长（Maya）的消费取代私域基线」，其余家长只在背景里出现；机制上等价于 **D**（`E_closest_mimicry`），与 gold **A**（`A_peer_weighted`）要求的**多源亲疏加权聚合**相反。

#### Qwen

`parsed_letter` → `parsed_rule_id`：**D** → **E_closest_mimicry**

```text
Situation: The protagonist faces peer pressure after their close friend Maya spent significantly more on a lavish party, while others' budgets cluster near their original plan. Key relationships and social visibility drive the decision.  
Psychology: Torn between personal frugality and Maya's influence, the protagonist weighs intimacy against group norms, feeling stronger pull from their closest tie despite broader moderation.  
Choice_reason: Rule D aligns with prioritizing the closest relationship (Maya's choice) without averaging in peripheral influences, matching the emotional weight of their bond.

Choice: D
```

**简析**：明确写出「最亲密纽带优先、不把外围平均进来」；与 **D**（`E_closest_mimicry`）一致，却把「亲疏」理解成**只保留最强边**，未对多名家长赋权再合成，故与 **A**（`A_peer_weighted`）错开。

#### gpt-4

`parsed_letter` → `parsed_rule_id`：**D** → **E_closest_mimicry**

```text
The protagonist values their close relationship with Maya, who is their college roommate turned mom-confidante, and Maya's party plans are clearly extravagant. Choosing the rule that mirrors Maya's plans directly aligns with the protagonist’s relational closeness to her. Thus, adopting Maya's exact budget reflects the importance of this intimate relationship.

Choice: D
```

**简析**：结论句是「规则应让预算**一对一镜像** Maya」；与 **D** 同型，是 mimic 与加权聚合的经典混淆。

#### ollama-llama3-1-latest

`parsed_letter` → `parsed_rule_id`：**B** → **C_uniform_avg**

```text
Situation: The protagonist has budgeted $3,486 for a backyard party but must decide whether to increase it due to social pressure from the group chat, especially Maya who is close to them, and the varying extravagance of others.

Psychology: The protagonist is likely feeling swayed by the group's suggestions and Maya's influence, which could lead them to compromise their private baseline for a "matching" experience, but they seem hesitant about how far to go in keeping up with the group.

Choice_reason: Choice B is the most fitting rule because it acknowledges the peer component through an unweighted mean, striking a balance between social pressure and personal preference without being overly influenced by one particular close tie or the single most expensive option.

Choice: B
```

**简析**：`Choice_reason` 明确选 **B** 的理由是「同伴项用**未加权均值**、在单条最贵与单条最亲之间取平衡」——这正是本题 **B**→`C_uniform_avg`（匀权平均、抹平亲疏梯度），与 gold **A** 的 proximity-weighted 合成相反；相较旧跑分，小 Llama 这里**叙事与字母自洽**。

#### ollama-llama3-8b

`parsed_letter` → `parsed_rule_id`：**C** → **B_top_anchor**

```text
- Situation: The scenario highlights the pressure to match Maya's extravagant party, while close ties like my college roommate and background acquaintances like Ben and Chloe influence my decision. As a parent, I want to prioritize relationships with other parents.
- Psychology: I'm torn between my private plan and the social pressure from close friends like Maya, who will likely judge my party as ordinary. My desire for social harmony may lead me to compromise on my budget.
- Choice_reason: Rule (C) aligns with my psychology because it considers the most relationally close parent, Maya, and layers her influence on top of my private floor. I might upgrade the pony to match her or stick to my quiet plan.

Choice: C
```

**简析**：三行模板把社会拉力收成「**Maya 这一最亲家长**叠在私域之上」；与 **C**→`B_top_anchor` 一致，Ben／Chloe 等仅作陪衬，未做多源加权。

#### ollama-qwen2-7b

`parsed_letter` → `parsed_rule_id`：**C** → **B_top_anchor**

```text
Situation: I'm considering upgrading my budget for a first birthday party based on the plans of others, specifically Maya who has planned an extravagant event that exceeds my initial $3,486. Others like Ben and Dave have slightly lower budgets but are also moving towards higher spending.

Psychology: I feel social pressure from comparing my budget to more extravagant plans and might be persuaded by the allure of matching Maya's spending. However, my private floor prefers a less flashy party that reflects my values on simplicity and affordability.

Choice_reason: The rule that best explains my potential decision is one that acknowledges both social influence (from close relationships) and maintains consideration for personal boundaries (my initial budget). Therefore, I choose to focus on the peer component of closest parents' decisions rather than adopting the most extravagant plan.

Choice: C
```

**简析**：口头上要「closest parents' decisions」而非最奢一档，仍把社会项压到**最熟关系**上；选 **C** 与 `B_top_anchor` 一致，与 **A** 的多边加权不符。

#### ollama-qwen2.5-14b

`parsed_letter` → `parsed_rule_id`：**D** → **E_closest_mimicry**

```text
Situation: Maya's extravagant budget overshadows others, making the protagonist question if sticking to her planned backyard bash fits in. Close friends and acquaintances show varying levels of expenditure.
Psychology: The protagonist feels pressure to match close friend Maya but is hesitant to deviate too much from their initial vision due to financial constraints.
Choice_reason: The rule suggests upgrading based on the most intimate friendship, pushing closer to Maya's extravagant budget

Choice: D
```

**简析**：短模板与三路大模型同构：压力来自 Maya 的奢侈、升级理由写「**最亲密友谊**」；选 **D**（`E_closest_mimicry`）与叙事一致，说明 **14B 本地模型**在此题上也能稳定落到单锚错型而非乱选字母。


---

## 2. `evalA__d1_cele_02__alpha_high__disp_high__skew_low__01`

`scene_id`：**d1_cele_02** · `cell_id`：**alpha_high__disp_high__skew_low** · 错选直方图：`F_median_anchor`×3, `D_pure_private`×2, `E_closest_mimicry`×2 · 众数：**`F_median_anchor`**

**Gold**：`B` → `A_peer_weighted`。

| 字母 | `parsed_rule_id` |
|---|---|
| `A` | `E_closest_mimicry` |
| `B` | `A_peer_weighted` |
| `C` | `F_median_anchor` |
| `D` | `D_pure_private` |

**题干**（摘自 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中该 `task_id` 的 `prompt` 字段，与评测时模型所见一致）：

```text
Scenario:
Our parenting group chat buzzed as I planned Noah's first birthday. My initial gut said we could throw a sweet backyard gathering for $931 — cupcakes, balloons, maybe a borrowed bouncy castle. But scrolling through our thread, memories of last month's bashes flashed by. My best friend Maya, who hosted our baby group since day one, dropped photos of her son's carnival-themed extravaganza costing $1,361 complete with face painters and custom cookies. Then there was Mark, whose toddler I only recognize from Instagram reels, mentioning his $1,013 spend on a petting zoo rental. Lisa, another face from the welcome meetup last year, shared her $1,038 princess tea party setup in the group album. My phone chimed with Maya asking about our plans while Mark’s lavish balloon arch smirked from my feed. That simple $931 vision now felt flimsy against their glittering proof of love.

Candidate rules for party budget:
(A) Lean fully on the closest-in-life parent's commitment and make that the sole number for the party budget — at $1,361.
(B) Let each of the parents tug the party budget in proportion to how much weight that relationship carries, layered above the baseline need — near $1,979.
(C) Take the point where half of the parents land above and half below, then lift the baseline by that central value — near $1,777.
(D) Stick with what the private baseline dictates without any lift from what the circle is doing — the party budget sits at $931.

Output format (STRICT — do not add extra lines):
Reasoning: <exactly 1 to 3 sentences, no more>
Choice: <one letter: A, B, C, or D>
```


### 各模型输出

#### DeepSeek-V4-Flash

`parsed_letter` → `parsed_rule_id`：**A** → **E_closest_mimicry**

```text
Situation: The protagonist's initial modest plan feels inadequate after seeing higher spending by other parents, especially best friend Maya's lavish party, creating social pressure to match perceived norms.

Psychology: The protagonist values Maya's friendship most and feels her own plan is flimsy against the group's displays, leading her to prioritize aligning with Maya's commitment over others.

Choice_reason: The rule that leans solely on the closest parent's spend fits because Maya's influence is strongest and directly mentioned, overriding the baseline.

Choice: A
```

**简析**：全文把决策压成「**只跟 Maya 这位最亲家长的支出对齐**」，与 **A**→`E_closest_mimicry` 一致；题干里 Mark／Lisa 等仍被叙事边缘化，与 gold **B**（`A_peer_weighted`）的多源加权不符。

#### Qwen

`parsed_letter` → `parsed_rule_id`：**C** → **F_median_anchor**

```text
Okay, let's try to work through this. The scenario is about planning a birthday party budget and feeling pressured by what others in the parent group have spent. The protagonist initially thought $931 was enough, but after seeing friends' extravagant parties, they feel their plan is too modest.

Looking at the candidate rules, each one represents a different way to adjust the budget based on social influence. Rule A says to match the closest friend (Maya) who spent $1,361. Rule B suggests a weighted average based on relationship importance, leading to a higher amount. Rule C uses the median of the group's spending and adds that to the baseline. Rule D sticks to the original $931 regardless of others.

The situation is that the protagonist is influenced by their close friends (Maya, Lisa) and others they know less well (Mark). The key here is the social pressure from the group, especially the closest relationships, which might push the budget higher. The protagonist's psychology is caught between their own budget and wanting to keep up with the group's expectations. 

Rule A focuses solely on the closest friend's spending, which might be tempting if Maya's opinion is highly valued. But the scenario mentions other people too, like Mark and Lisa, whose higher spending could contribute to feeling inadequate. However, since Maya is a best friend and the host of the group, her influence might be the strongest. But the answer options might require considering more than one. Wait, Rule D is sticking with the original, but the protagonist feels their plan is flimsy now. So they might be swayed. Rule C takes the median, which would be around $1,038 (sorted: 1013, 1038, 1361; median is 1038), then adding baseline 931? Wait the median is $1,038, and then the rule says to lift the baseline by that central value. Wait, original baseline is $931. The rule says "take the point where half are above/below... then lift the baseline by that central value." So adding 931 + 1038? That would be 1969, but choice C says near 1777. Wait 1038 is the median. Maybe 1038 is the central value, so adding to the baseline? Wait the baseline is 931. So 931 + 1038 = 1969, but the answer C says near 1777. Hmm maybe I miscalculated. Let me check the given numbers. The four mentioned costs are Maya's $1361, Mark's $1013, Lisa's $1038, and the protagonist's $931. Wait no, the protagonist's initial plan is $931. The three others are Maya (1361), Mark (1013), Lisa (1038). So the four people in the group, including the protagonist, have 931,1013,1038,1361. To find the median of the three others (since the protagonist's own isn't part of the group's data?), or all four including? Scenario says "half of the parents land above and half below" — maybe the group includes the protagonist? Wait, but the protagonist is deciding their own budget. The other parents mentioned are Maya, Mark, Lisa — three. So the group's other parents' numbers are three: 1361, 1013, 1038. The median of those is 1038. Rule C says take that median and lift the baseline by that. So adding 931 + 1038 = 1969? But the given option C says near 1777, which doesn't match that. Wait maybe the rule is to set the budget to the median instead of adding? Or perhaps the numbers are arranged differently. Alternatively, maybe the central value is the median of all four numbers, but the protagonist's 931 would be below the median of the others. Maybe I'm misunderstanding the rule. Alternatively, perhaps the median of the three others is

# --- 评测补写：raw 无 Choice 行，按 parsed_letter 补 ---
Choice: C
```

**简析**：长稿在中位数与备选项金额间**自我打架**，稿末截断后 **C** 为评测补写；整体更像被「中位／折中」话术拖向 **C**（`F_median_anchor`），而非对 gold **B** 的 proximity-weighted 论证。

#### gpt-4

`parsed_letter` → `parsed_rule_id`：**D** → **D_pure_private**

```text
The protagonist feels pressure from the group to raise their budget, but ultimately the decision should reflect their personal comfort and financial priorities. They initially planned for $931, which indicates a private baseline that aligns with their means and intentions. Sticking to their original plan reflects independence and genuine expression of love without succumbing to external pressure.

Choice: D
```

**简析**： normative 口吻主张「**舒适与优先级 > 从众**」，与 **D**→`D_pure_private` 一致；社交压力被当作应被压制的噪声，与 **B** 的加权社会项相反。

#### ollama-llama3-1-latest

`parsed_letter` → `parsed_rule_id`：**A** → **E_closest_mimicry**

```text
Situation: The protagonist's decision is influenced by the backyard gathering plan they initially envisioned, but also by the extravagant parties shared in the group chat by close friends and acquaintances.

Psychology: The protagonist feels pressure to keep up with their social circle's spending habits, and this pressure seems to override their initial preference for a more modest celebration.

Choice_reason: The protagonist is swayed by Maya's example since Maya has been hosting the baby group and their relationship carries significant weight, suggesting they will lean heavily on Maya's commitment as the sole number for the party budget.

Choice: A
```

**简析**：`Choice_reason` 把 Maya 的支出写成**唯一预算锚**；选 **A**（`E_closest_mimicry`）与叙事一致，仍是单点模仿而非 gold **B**。

#### ollama-llama3-8b

`parsed_letter` → `parsed_rule_id`：**D** → **D_pure_private**

```text
- Situation: As I plan Noah's first birthday, the peer pressure from fellow parents' extravagant parties weighs heavily, making me question my initial $931 budget. My close relationships with these parents are at stake.
- Psychology: I'm torn between sticking to my private baseline and being influenced by others' displays of love; social pressure makes it hard for me to commit to a lower budget when everyone else is spending more.
- Choice_reason: Since I'm hesitant to compromise on my initial budget while also wanting to match the spirit of love and generosity shown by others, I'll opt for a middle ground that considers the peer pressure without completely abandoning my private baseline.

Choice: D
```

**简析**：`Choice_reason` 写「middle ground、不放弃私域基线」，与 **D**→`D_pure_private`（**零社会拉力**）在机制上**矛盾**；更像想选折中或中位，却落笔为 **D**，属于**叙事与字母错位**。

#### ollama-qwen2-7b

`parsed_letter` → `parsed_rule_id`：**C** → **F_median_anchor**

```text
Situation: The protagonist, amidst planning Noah's birthday party, is influenced by her close friends' extravagant celebrations and their social media posts, which contrast with her initial budget of $931.

Psychology: The protagonist feels the pressure to match or exceed the spending levels of her well-connected friends, fearing she might appear less committed as a parent compared to these individuals. Her perceived value in this friendship group is tied to aligning closely with their collective expectations for such celebratory events.

Choice_reason: This scenario highlights the protagonist's need to strike a balance between adhering to her personal budget and aligning with social pressures from close friends. The rule that best encapsulates this dynamic is one that considers the impact of peer influence on spending decisions but maintains a baseline of self-determined needs, thus making choice (C) "Take the point where half of the parents land above and half below, then lift the baseline by that central value", fitting most closely as it represents an equilibrium between the individual's decision-making process influenced by social comparison and her own private budgetary considerations.

Choice: C
```

**简析**：显式引用备选项 **C** 的中位表述，把「同伴压力 vs 自决基线」收成**中心值抬升**；与 **C**→`F_median_anchor` 自洽，但未按亲疏对 Maya／Mark／Lisa **分别赋权**（与 gold **B** 相背）。

#### ollama-qwen2.5-14b

`parsed_letter` → `parsed_rule_id`：**C** → **F_median_anchor**

```text
Situation: Noah's birthday party budget decision hinges on comparing the protagonist's original modest plan to other parents' lavish expenditures, influenced by social dynamics and expectations within a close-knit parenting group.
Psychology: The protagonist feels pressure from friends sharing expensive celebrations but struggles aligning with such spending while staying true to personal financial comfort zones.
Choice_reason: Considering both friendship bond strength and peer influence, balancing between baseline personal preference and relational societal norms nears $1,777.

Choice: C
```

**简析**：三句模板把结论绑到「**基线 + 关系性规范之间折中 → 接近 $1777**」；与 **C**（中位锚类）弱匹配，仍无**显式 g_ij 加权**推演，但比 7B 更少出现字母与一句话完全脱节。


---

## 3. `evalA__d1_cele_02__alpha_high__disp_low__skew_low__00`

`scene_id`：**d1_cele_02** · `cell_id`：**alpha_high__disp_low__skew_low** · 错选直方图：`H_equal_mix`×5, `D_pure_private`×2 · 众数：**`H_equal_mix`**

**Gold**：`A` → `A_peer_weighted`。

| 字母 | `parsed_rule_id` |
|---|---|
| `A` | `A_peer_weighted` |
| `B` | `H_equal_mix` |
| `C` | `D_pure_private` |
| `D` | `G_counter_conformist` |

**题干**（摘自 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中该 `task_id` 的 `prompt` 字段，与评测时模型所见一致）：

```text
Scenario:
Sitting in our young-parents group chat, planning my toddler's first birthday bash, I can't help but scroll through all the lavish past parties on Instagram—balloon arches, bouncy castles, the works. My friend Sarah, a fellow mom from the neighborhood, shared her budget: $4,332 for last month's event, complete with a mini-zoo. Then Mark, another regular in our chat, chimed in about his $4,750 splurge on a celebrity-style photographer. Emma, who I see at daycare drop-offs, recalled her $4,123 spend for gourmet catering. And Jake, a dad from the block, posted his detailed $3,771 breakdown with pride. Honestly, if I were just throwing this for ourselves without anyone watching, I'd keep it simple and cozy at $3,510—just cake, close family, and backyard fun. But with everyone flaunting their numbers online, I'm torn: go modest like my gut says, or match up to avoid looking cheap.

Candidate rules for party budget:
(A) Stack the baseline need first, then layer a peer lift where each parent's share tracks their standing in the life — arrives near $7,117.
(B) Split the difference evenly between the private floor and the closeness-weighted social pull, without scaling by how much peers matter — around $3,884.
(C) Ignore the parents altogether and commit to exactly the level that the private cost-and-need calculation would justify — about $3,510.
(D) React against the parents' visible trend by pulling the party budget strictly below the private floor — around $0.

Output format (STRICT — do not add extra lines):
Reasoning: <exactly 1 to 3 sentences, no more>
Choice: <one letter: A, B, C, or D>
```


### 各模型输出

#### DeepSeek-V4-Flash

`parsed_letter` → `parsed_rule_id`：**B** → **H_equal_mix**

```text
Situation: The protagonist sees peers spending $3,771–$4,750 on toddler parties, while her private preference is $3,510. Social comparison pressure and fear of appearing cheap drive the dilemma.

Psychology: She feels torn between her own modest desire and the urge to match peers' visible spending, weighing relationships with nearby parents like Sarah, Mark, Emma, and Jake.

Choice_reason: Splitting the difference between her private floor and the social pull balances her true preference with peer influence without overreacting.

Choice: B
```

**简析**：`Choice_reason` 把机制写成**私域 floor 与社会拉力对半分**；与 **B**→`H_equal_mix`（固定比例混合、不显式用亲疏矩阵）一致，与 gold **A**（`A_peer_weighted`）相背。

#### Qwen

`parsed_letter` → `parsed_rule_id`：**B** → **H_equal_mix**

```text
Okay, let's tackle this. The protagonist is planning their toddler's first birthday and is torn between their own budget of $3,510 and the pressure to spend more after seeing others' expensive parties. The key here is the social dynamics with friends like Sarah, Mark, Emma, and Jake, who have all spent over $3,700. The options are rules A to D.

First, the situation: The parent feels social pressure from friends in the chat group and daycare, who have spent significant amounts. The options need to address whether they bow to peer pressure or stick to their own plan.

Psychology: The protagonist's internal conflict is between wanting to keep things simple (private need) and fearing judgment (social pressure from peers). They might not want to feel like they're skimping compared to others.

Now, the rules. Rule C suggests sticking to the private cost, which is $3,510. Rule B is averaging the private and social factors, leading to around $3,884. Rule A is大幅提升 the budget way above others, which seems extreme. Rule D is going below the private floor, which might be too rebellious.

The protagonist seems to want to resist the pressure but is concerned about appearing cheap. The scenario mentions "torn" but their initial inclination is to go modest. They interact regularly with these peers (neighborhood, daycare, group chat), so social standing matters. However, if they choose C, they commit to their own budget despite peer influence, which aligns with not letting others' displays dictate their decision. But maybe the fact that they are comparing and considering the others' spending suggests some middle ground. Wait, Rule B splits the difference between their own budget and the social pull. However, the answer might hinge on whether the protagonist wants to avoid looking cheap but still stay relatively modest. The private floor is $3,510, and others are around $4k. Splitting difference evenly without scaling by peer influence might land at $3,884 (maybe average of $3,510 and maybe the average of others' numbers? Let's calculate: average of the friends' spending: 4332 +4750+4123+3771. Let's add: 4332+4750=9082; 4123+3771=7894. Total 9082+7894=16976. Divide by 4: 4244. So average of friends is ~$4,244. If the private floor is 3510, the midpoint between that and 4244 is (3510+4244)/2 = 3877, which is close to $3,884 as in B. So if the protagonist chooses B, they're compromising. However, the question is what the protagonist would commit to. If they are feeling torn but don't want to overextend, maybe they go for the middle ground rather than outright ignoring peers (C) or over-the-top (A). Alternatively, the text says "if I were just throwing this for ourselves without anyone watching, I'd keep it simple and cozy at $3,510". So the protagonist's true preference is C but feels pressure. However, the psychology might lead them to a compromise, hence B. But the question is which rule they would commit to. The rules are a bit abstract. Rule B says split the difference evenly, not scaling by peer importance. Rule C is ignoring peers entirely. The protagonist's hesitation is because of the desire to not look cheap compared to others. So if they go with B, they balance their own needs and social expectations. Alternatively, if they are determined to not let others influence beyond necessity, maybe C. But the scenario says the protagonist is torn, so the answer might be B. But need to check again.

The protagonist's own budget is below all peers' except Jake's $3771. The protagonist's value is $3,510, which is the lowest. The others are in the 3771-4750 range. The average of friends' spending is around $4,244. So

# --- 评测补写：raw 无 Choice 行，按 parsed_letter 补 ---
Choice: B
```

**简析**：长稿用好友支出均值与私域中点自洽地论证选项 **B**；文末 **Choice** 为评测补写接上。错因仍是「会算无亲疏梯度的折中」，不会选 **A**。

#### gpt-4

`parsed_letter` → `parsed_rule_id`：**B** → **H_equal_mix**

```text
The protagonist feels pressure from social comparisons and wants to avoid appearing cheap among peers, but the $3,510 budget reflects their actual preference for a cozy event. Thus, splitting the difference allows them to balance personal desire with social influence while not overspending excessively.

Choice: B
```

**简析**：一句「splitting the difference」直接对应 **B**→`H_equal_mix`；同伴未按 Sarah／Mark 等**分别加权**。

#### ollama-llama3-1-latest

`parsed_letter` → `parsed_rule_id`：**B** → **H_equal_mix**

```text
Situation: The protagonist is torn between their private preference for a modest party and the pressure to match the lavish spending of peers, all while being transparent about expenses on social media.

Psychology: The protagonist's decision is driven by a desire to avoid appearing cheap in front of friends and fellow parents online, despite their initial inclination towards simplicity.

Choice_reason: The protagonist wants to achieve a balance between fitting in with others and staying true to their own values; thus, they will choose the middle ground option that accommodates both social pressure and private preference.

Choice: B
```

**简析**：`middle ground` 与 online 攀比并置，选 **B**→`H_equal_mix` 与「无梯度折中」叙事一致。

#### ollama-llama3-8b

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
-Situation: The protagonist is torn between their private preference for a simple and cozy party and the pressure to match others' extravagant celebrations, feeling like they're being judged by their peers.

-Psychology: Social media amplifies the pressure to keep up with others, making the protagonist question whether to prioritize authenticity or conformity.

-Choice_reason: The protagonist is motivated by the desire not to appear cheap, but also wants to celebrate responsibly; rule (C) best captures this tension between private need and social lift.

Choice: C
```

**简析**：`Choice_reason` 强调「私域需要 vs 社会拉升」的张力并点名 **rule (C)**，而本题字母 **C** 映射 **`D_pure_private`**（非折中项）；属于**用「冲突描写」勾到私域字母**，与 gold **A** 相背，且叙事与「纯私域」字面定义略拧。

#### ollama-qwen2-7b

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
Situation: The protagonist feels torn between keeping a modest party budget for personal preference versus matching the extravagant budgets shared by friends online to avoid being perceived as cheap.

Psychology: She experiences social pressure from her peers' online displays of spending on their children's birthday parties, which makes her question whether she should aim for a similarly high cost.

Choice_reason: The protagonist's decision-making process is influenced by external social comparisons and peer expectations.

Choice: C
```

**简析**：通篇写**攀比与犹豫**，未论证「零社会拉力」；选 **C**→`D_pure_private` 更像在弱信号下**用私域字母收束**，与 gold **A** 相背。

#### ollama-qwen2.5-14b

`parsed_letter` → `parsed_rule_id`：**B** → **H_equal_mix**

```text
Situation: The protagonist faces social pressure to spend extravagantly on a child's birthday while balancing a personal preference for modesty, influenced by the visible spending of close community members.
Psychology: She feels torn between her desire to maintain respect and inclusion within the group and her inclination toward simplicity, weighing carefully both peer influence and personal values.
Choice_reason: Balancing social expectations with personal comfort suggests she will moderately adjust her budget upwards from her private baseline to feel included without overextending.

Choice: B
```

**简析**：「**适度上调私域基线、融入但不透支**」与 **B**→`H_equal_mix` 同型；**14B** 在此题与多数大模型一样走**固定折中**而非加权聚合。


---

## 4. `evalA__d1_cele_02__alpha_high__disp_mid__skew_mid__00`

`scene_id`：**d1_cele_02** · `cell_id`：**alpha_high__disp_mid__skew_mid** · 错选直方图：`B_top_anchor`×3, `F_median_anchor`×3, `C_uniform_avg`×1 · 众数：**`B_top_anchor`**

**Gold**：`A` → `A_peer_weighted`。

| 字母 | `parsed_rule_id` |
|---|---|
| `A` | `A_peer_weighted` |
| `B` | `F_median_anchor` |
| `C` | `B_top_anchor` |
| `D` | `C_uniform_avg` |

**题干**（摘自 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中该 `task_id` 的 `prompt` 字段，与评测时模型所见一致）：

```text
Scenario:
Our young-parent group chat lit up with first-birthday plans, Instagram feeds still glittering from last month's bashes. My best friend Mia, who hosted my baby shower, went all out with a professional setup costing $10,014 — her photos flooded our chat. Jake, the dad from our parenting class, shared his backyard carnival expenses at $5,620. Then came acquaintances: Chloe, whom I only know through shared reels, posted a $5,223 garden tea party, while Derek, a sporadic commenter, revealed his drone-captured bash at $6,798. Privately, I'd penciled in $3,573 for a simple park gathering with homemade treats. But scrolling through their lavish spreads — balloon arches and custom cakes taunting me from the screen — I clutch my phone, wondering how thin my budget will stretch before the group notices.

Candidate rules for party budget:
(A) Anchor on the private floor, then shift toward the parents in proportion to how close each one really is — the resulting party budget sits around $11,051.
(B) Route the peer pull through the representative central parent and stack it on the private floor — the party budget lands near $9,439.
(C) Use the most tight-knit parent's move as the only relevant pull above the baseline need — the party budget ends up around $13,033.
(D) Pool the parents' visible choices with identical weights, then stack that on the private floor — lands near $10,104.

Output format (STRICT — do not add extra lines):
Reasoning: <exactly 1 to 3 sentences, no more>
Choice: <one letter: A, B, C, or D>
```


### 各模型输出

#### DeepSeek-V4-Flash

`parsed_letter` → `parsed_rule_id`：**C** → **B_top_anchor**

```text
Situation: The protagonist initially budgets $3,573 for a simple park party but sees lavish spending from Mia (best friend), Jake, Chloe, and Derek in the group chat, creating social pressure. The key factor is relational closeness.

Psychology: The protagonist feels torn between their private baseline and the desire to fit in, especially with best friend Mia, whose party stands out as the most personally significant influence.

Choice_reason: Rule C isolates the most tight-knit parent (Mia) as the sole relevant pull above the baseline, matching the protagonist's likely focus on that close relationship.

Choice: C
```

**简析**：`Choice_reason` 把 **C** 说成「**只取最 tight-knit 的 Mia** 作为唯一社会拉力」；即 **B_top_anchor**／单锚坍缩，与 gold **A**（`A_peer_weighted`）对多名同伴加权整合相反。

#### Qwen

`parsed_letter` → `parsed_rule_id`：**C** → **B_top_anchor**

```text
Okay, let's try to figure this out. The protagonist is in a parent group where everyone's sharing their expensive birthday parties. Mia, her best friend, spent over $10k. Jake, from parenting class, did a backyard carnival for $5k+ and others spent even more. The protagonist planned a cheaper park event with homemade stuff, around $3,500. Now she feels pressured looking at others' parties.

The candidate rules are about how much to adjust her budget based on others. Let's look at each option. 

Option A says anchor on her private floor but shift towards parents based on closeness. Since her friends are closer, their higher spending would push her budget up. Her original $3,573 might go up, but how much? If Mia is the closest, maybe her influence is stronger here.

Option B routes peer pull through a central parent and adds to her baseline. Maybe they take an average of the close parents and add to her original budget. If the central parent's spending is higher, this could push her budget higher than A?

Option C uses only the closest parent's spending over the baseline. If Mia is the closest, and she spent $10,014, then the protagonist might have to add that difference. Her original was $3,573. Mia's amount is $10,014, so the difference is $6,441. Adding that to baseline? Wait, but C says "only relevant pull above baseline need". Maybe she would mirror Mia's spending? Then her budget would jump to Mia's $10k, but the result here is $13k. Hmm, maybe Mia's plus something else?

Option D averages all parents' visible choices equally and adds that to her baseline. So averaging all the amounts: Mia $10k, Jake $5,620, Chloe $5,223, Derek $6,798. Average of those four is (10014 + 5620 +5223 +6798)/4 = let's see: 10014 + 5620 is 15634, plus 5223 is 20857, plus 6798 is 27655. Divide by 4 is ~6913.75. Then adding her baseline $3,573? Total around 10,486, which is close to D's $10,104 (maybe rounded differently). So D's result is around that. 

The protagonist is feeling the pressure from the group, especially close friends. Mia, who is her best friend, spent the most. She might feel more compelled to match Mia's spending more than others. So looking at the rules:

Option C uses the most tight-knit parent's move. If Mia is the closest, then her spending would dictate the adjustment. The result is around $13k. But the protagonist's original was lower. However, if the rule is that you take the closest parent's amount and add to baseline, maybe the $13k includes that. 

Option A says anchor on private floor then shift towards parents proportionally based on closeness. If Mia is very close and others less so, maybe A's result is $11k. But the protagonist's private baseline is $3,573, so adding Mia's higher amount but also others... but in what proportion. 

The scenario mentions that the protagonist is comparing her planned budget to others, especially Mia's high spending. The pressure is from her close friend, so she might feel the need to follow Mia's lead more than others. 

Between C and A. Option C uses only the closest parent's pull above baseline. If the protagonist's baseline is $3,573 and she's mirroring Mia's $10k, then the total would be Mia's amount. But the option C result is $13k, which is higher than Mia's $10k. Wait, maybe the rule is to add the difference from the closest parent's spending. So baseline plus (Mia's amount minus baseline) would result in Mia's numbers. But if it's stacking, maybe base plus the pull.

# --- 评测补写：raw 无 Choice 行，按 parsed_letter 补 ---
Choice: C
```

**简析**：长篇在 **A**（按亲疏比例位移）与 **C**（单 closest pull）之间权衡后仍收束到 **C**；**Choice** 为评测补写。整体仍是「**Mia 压强最大 → 单锚**」路径，与 gold **A** 相悖。

#### gpt-4

`parsed_letter` → `parsed_rule_id`：**C** → **B_top_anchor**

```text
The protagonist is most influenced by their close relationship with Mia, who hosted the baby shower, suggesting Mia's party budget would be the strongest pull above the baseline. Thus, the protagonist would likely choose to mirror Mia's approach, given the significance of their friendship.

Choice: C
```

**简析**：与 DeepSeek 同构：Mia／baby shower 被写成**唯一强 pull**，选 **C**→`B_top_anchor` 自洽。

#### ollama-llama3-1-latest

`parsed_letter` → `parsed_rule_id`：**B** → **F_median_anchor**

```text
Situation: The protagonist's friend Mia hosted an expensive party, and they're considering how to budget for their own child's first birthday, feeling pressure from social media to match or exceed friends' spending.

Psychology: The protagonist is being pulled by peer influence, but also wants to be mindful of their private financial constraints and stick to a simple plan – this is a delicate balance between seeking social validation and staying true to their personal budget.

Choice_reason: The protagonist is looking for a rule that accounts for the peer pull through friends, especially close ones like Mia, while still respecting their own baseline need; option (B) allows them to route the peer influence through a central figure in this context –

Choice: B
```

**简析**：`Choice_reason` 口头上把 **(B)** 说成「经 **central figure** 路由同伴拉力」，易与「中位／代表点」混淆；本题 **B** 字母映射 **`F_median_anchor`**，但正文**未给出中位数或中位锚的实质推导**，与标签贴合度一般。

#### ollama-llama3-8b

`parsed_letter` → `parsed_rule_id`：**B** → **F_median_anchor**

```text
- Situation: As I worry about my party budget being under scrutiny, the social pressure from extravagant displays and peer comparison takes center stage. My private baseline of $3,573 is threatened by the group's lavish parties.
- Psychology: I feel a strong desire to fit in while also wanting to remain true to myself, making social pressure an important factor in this decision. The close relationships with Mia and Jake make their party budgets more influential than those of acquaintances like Chloe and Derek.
- Choice_reason: Since the peer pull is significant due to the desire to fit in, and the representative central parent (Jake) has a moderate budget ($5,620), I will use rule (B) that pools the parents' visible choices with identical weights.

Choice: B
```

**简析**：前半承认 Mia／Jake 更近，`Choice_reason` 却把 rule (B) 说成以相同权重池化所有家长——这一定义更接近 **D**（`C_uniform_avg`），与 **B**（本题映射为 `F_median_anchor`）严重错位，属小模型常见的字母与机制张冠李戴。

#### ollama-qwen2-7b

`parsed_letter` → `parsed_rule_id`：**B** → **F_median_anchor**

```text
Situation: The protagonist compares lavish birthday party expenses shared by friends and acquaintances against their own plan for a simple, homemade gathering at the park.

Psychology: Feeling social pressure from extravagant celebrations, she contemplates her budget, considering how closely each friend is in reality versus online portrayals of their connections.

Choice_reason: Given her private inclination towards simplicity aligned with current financial constraints compared to her perceived need to match or exceed the lavish spending of more connected friends, rule (B) best explains that she might adjust her plan closer to what these peers can afford while still considering the strength of her real relationships.

Choice: B
```

**简析**：提到亲疏与现实／线上形象，但收束到 **rule (B)** 时仍**无中位演算**；与 **B**→`F_median_anchor` 弱匹配，与 gold **A** 更远。

#### ollama-qwen2.5-14b

`parsed_letter` → `parsed_rule_id`：**D** → **C_uniform_avg**

```text
Situation: The protagonist faces social pressure to match others' extravagant first-birthday celebrations, balancing her desire for a modest gathering against the expectations of variously close friends and acquaintances.
Psychology: She feels compelled to consider closer relationships but also notes peripheral influencers, suggesting she might adjust her budget upward significantly based on perceived norms rather than strict financial limits.
Choice_reason: Considering both direct friendships and broader peer influence, she likely aims for a middle ground that respects her budget while not appearing frugal in comparison —

Choice: D
```

**简析**：`Choice_reason` 写 **middle ground**，却选 **D**→`C_uniform_avg`（全员匀权平均）；**叙事偏折中、字母偏匀权**，与 gold **A** 均不符，贴合度弱于本题其他输出。


---

## 5. `evalA__d1_dining_03__alpha_high__disp_low__skew_high__01`

`scene_id`：**d1_dining_03** · `cell_id`：**alpha_high__disp_low__skew_high** · 错选直方图：`D_pure_private`×6, `C_uniform_avg`×1 · 众数：**`D_pure_private`**

**Gold**：`B` → `A_peer_weighted`。

| 字母 | `parsed_rule_id` |
|---|---|
| `A` | `C_uniform_avg` |
| `B` | `A_peer_weighted` |
| `C` | `D_pure_private` |
| `D` | `B_top_anchor` |

**题干**（摘自 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中该 `task_id` 的 `prompt` 字段，与评测时模型所见一致）：

```text
Scenario:
After our consulting firm's promotion announcements, Jake—my teammate from the Chicago project—slid into our booth at the afterparty. 'Priya and I found this insane twelve-course spot,' he grinned, tapping his phone. 'Reservation's $811 a head, but worth it for the truffle caviar.' Across the table, Priya nodded, swirling her drink. 'My splurge hit $1,205 last month. Treat yourself, right?' Our group chat blew up: Leo, who I only know from the London office Slack, posted '$656? Cheap for Michelin-starred.' Then Ben, an analytics guy I met once at a conference, chimed in: '$834. Book fast.' Sam, lurking with a '$601' emoji reaction, felt like a ghost from the welcome mixer. I stared at my own draft reply, fingers hovering. Privately, I’d budgeted just $241—enough for my favorite neighborhood bistro. Now the team’s eyes were on me, Venmo links flooding the screen.

Candidate rules for monthly restaurant spend:
(A) Treat each friend as one democratic vote and scale by the group's flat average above the baseline — near $1,009.
(B) Use a proximity-graded blend of what the friends are doing as a lift above the personal need — the monthly restaurant spend sits near $1,087.
(C) Stick with what the private baseline dictates without any lift from what the circle is doing — the monthly restaurant spend sits at $241.
(D) Mute the group as a whole and route the peer term through the sole nearest-in-life friend's move above the baseline — near $1,367.

Output format (STRICT — do not add extra lines):
Reasoning: <exactly 1 to 3 sentences, no more>
Choice: <one letter: A, B, C, or D>
```


### 各模型输出

#### DeepSeek-V4-Flash

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
Situation: The protagonist faces social pressure from colleagues with high restaurant spends, but these peers are mostly peripheral acquaintances, not close friends, making the private budget of $241 salient.
Psychology: The protagonist prioritizes personal financial reality over group conformity, feeling no strong obligation to match distant peers' extravagant habits.
Choice_reason: The rule that sticks to the private baseline without any lift from the circle best reflects the protagonist's resistance to peer pressure.

Choice: C
```

**简析**：用「同事多为**外围熟人**、无义务跟风」把社会项关掉；与 **C**→`D_pure_private` 一致，与 gold **B**（`A_peer_weighted`）对核心同事加权相反。

#### Qwen

`parsed_letter` → `parsed_rule_id`：**A** → **C_uniform_avg**

```text
Okay, let me try to work through this. The scenario is about the protagonist facing peer pressure to spend more on a restaurant after their team's promotion. The group is pushing for a high-end place, with others in the group spending over $600 up to $1205. The protagonist's budget is only $241, which is much lower. They need to decide which rule to follow.

So looking at the options:

A) Average the group's spends which is around $1009. That would mean the protagonist should spend close to that average. But the group includes people they don't know well, like Leo from another office or Ben they met once. Maybe the average is skewed by people not close to them.

B) Proximity-graded blend, so considering how close each friend is and taking a weighted average, leading to $1087. This might weigh closer friends more, but the protagonist might not feel close to everyone in the group. Like Priya and Jake are close teammates, but others are more peripheral.

C) Stick to their own budget of $241 regardless of others. This is the private baseline. But the social pressure is strong here, with everyone else committing to high amounts. Would they feel left out if they don't join?

D) Use the nearest-in-life friend's spend, which is $1367. Wait, the scenario mentions Priya spent $1205, so maybe this 'nearest' friend is someone else not mentioned? Or maybe Jake or Priya is considered nearest. If D is based on the closest friend, then maybe Jake or Priya's amount is the lift. But Priya's is $1205, but option D's number is $1367. Hmm, maybe that's a different friend not in the example. Or perhaps Sam, but Sam reacted with $601. Not sure. The key here is whether the protagonist feels closest to someone who's spending the most.

The protagonist's dilemma is between social conformity and personal budget. They might feel pressure from close teammates (Jake and Priya) but also from others they barely know. The psychology would be torn between wanting to fit in with the team and sticking to their budget. However, since the group includes people they have varying levels of closeness to, the closest friends (Jake and Priya) are influencing them. 

Option C says to stick with their own baseline. But if the team is expecting them to join, maybe they feel compelled to at least consider others. Option D is the highest, so unless their closest friend is someone who's spending that much, which isn't clear. 

Alternatively, if the protagonist's nearest-in-life friend in the scenario is Priya or Jake, who spent $1205 and implied to book fast. So the sole nearest friend's move above baseline might be Jake's $811 or Priya's $1205. If that's the case, D would take that amount. However, the option's number is $1367, which isn't in the scenario. Wait, maybe the 'nearest' friend is not mentioned here. Maybe the protagonist has another friend outside the given scenario. Or perhaps the $1367 is derived from a combination? Hmm, maybe I'm misunderstanding.

Alternatively, the scenario mentions Sam with a $601 reaction. Maybe Sam is the nearest-in-life friend. But the reaction is $601, but the option D's value is $1367. That doesn't align. Wait, maybe the 'nearest' friend's move above baseline is calculated differently. Maybe the baseline here refers to the protagonist's $241, so $1367 would be $241 plus the lift from the nearest friend's move. Not sure.

Alternatively, maybe the answer is C because the protagonist might resist peer pressure and stick to their budget. But the situation where the team's eyes are on them and Venmo links are flooding in suggests strong social pressure. But which rule applies?

The key might be the relationships. The prompt mentions that Jake is a teammate from the Chicago project (so close colleague), Priya as well. Others are from different offices or met once. The protagonist may feel closer to Jake and Priya, but others are more peripheral. 

Rule B is a

# --- 评测补写：raw 无 Choice 行，按 parsed_letter 补 ---
Choice: A
```

**简析**：长稿在 **A**（群体均值）、**B**（亲疏加权，即 gold）、**C**（私域）之间来回，稿末截断于「Rule B is a」；**A** 由评测补写。即便落 **A**（`C_uniform_avg`），仍是**匀权聚合**而非按 **g_ij** 的亲疏加权。

#### gpt-4

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
The protagonist is likely to stick with their privately budgeted amount of $241, as it reflects their personal financial comfort and priorities, rather than succumbing to peer pressure. This approach would avoid overspending based on what others deem appropriate.

Choice: C
```

**简析**：明确「**$241 私域舒适区**、不因他人定义而超支」；与 **C**→`D_pure_private` 一致，与 **B** 相悖。

#### ollama-llama3-1-latest

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
Situation: The protagonist's private budget and the group's collective splurge create a situation of social pressure to match or exceed the expenses of their peers, while also staying true to personal financial goals.

Psychology: The protagonist seems aware of the social norms of the consulting firm and the pressure to participate in high-end dining experiences, but is hesitant due to their private baseline budget being significantly lower than their colleagues' plans.

Choice_reason: The protagonist's primary concern appears to be adhering to their set budget, suggesting they prioritize private needs over social pressures; therefore, a rule that prioritizes personal baseline spending would best guide the decision.

Choice: C
```

**简析**：`Choice_reason` 把结论写成「**优先个人基线支出规则**」；与 **C**→`D_pure_private` 一致。

#### ollama-llama3-8b

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
Situation: The protagonist's private baseline for monthly restaurant spend is $241, but the group's discussions and expectations are influencing their decision.

Psychology: The protagonist feels social pressure from close colleagues like Jake and Priya, as well as anonymous peripheral participants like Sam, which may lead them to depart from their personal budget.

Choice_reason: The protagonist doesn't want to deviate too far from their peers' spending habits while still staying true to their financial goals; this suggests they will gravitate towards a rule that lifts their spend without completely abandoning their baseline.

Choice: C
```

**简析**：`Choice_reason` 写「**适度抬升支出、又不放弃基线**」，听起来像折中或锚定；但 **C** 在本题映射 **`D_pure_private`**，与「仍有社会拉升」的叙述**不完全同向**，存在轻微信号错位。

#### ollama-qwen2-7b

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
Situation: After promotion announcements, Jake and Priya invite the team to an expensive restaurant, sparking discussions on costs; private budget at $241 contrasts with group's higher spending expectations.
Psychology: As a peer under pressure from friends who advocate for extravagant expenses, I feel the need to keep up social image while staying within my own limited personal baseline.
Choice_reason: Following personal needs and avoiding peer pressure without sacrificing social connection best aligns with maintaining autonomy over finances and not deviating from individual spending habits;

Choice: C
```

**简析**：在「维持自主、不偏离个人消费习惯」上收束；与 **C**→`D_pure_private` 一致，Jake／Priya 仅作压力背景。

#### ollama-qwen2.5-14b

`parsed_letter` → `parsed_rule_id`：**C** → **D_pure_private**

```text
Situation: The protagonist faces social pressure to match their team's high-end dining expenses despite personal budget constraints, navigating peer dynamics and financial comfort zones.
Psychology: Struggling with the contrast between personal finances and the apparent splurge of colleagues, they feel torn but ultimately assess that financial integrity trumps outward conformity in this context.
Choice_reason: Adhering closely to personal spending limits while silently observing group indulgences reflects a balance of social pressures without compromising financial health.

Choice: C
```

**简析**：先承认撕扯，再以「**财务诚信优于外向从众**」落锤；与 **C**→`D_pure_private` 一致。本题 **7** 路里有 **6** 路落到同一错型，说明在「晋升聚餐 + 高曝光」场景下模型极易**零社会项化**。


---

## 本地小参数四模型共性问题（更新）

以下指 **`ollama-llama3-1-latest`、`ollama-llama3-8b`、`ollama-qwen2-7b`、`ollama-qwen2.5-14b`**，与三路大模型对照；基于**本 Top-5 新题集**（七路全错子集收窄后重新贪心选出）。

**1. 叙事模板仍占主导**  
「私域 vs 攀比 / 合群」「撕裂 → 折中」「最亲一人压过全场」仍高频出现；**14B** 在题 1 能写出与 **D**（`E_closest_mimicry`）一致的短论证，但在题 4、题 5 仍常与 **7B／8B** 一样出现**短模板或论证截断**。

**2. 字母与机制错位未消失**  
题 2 **llama3-8b**（口头 middle ground、选项 **D** 私域）、题 3 **llama3-8b／7B**（张力描写 vs **C**→私域）、题 4 **llama3-8b**（写「identical weights」却选 **B**→中位）等，说明 **14B** 并未从根上消除「**口头一种规则、勾另一种字母**」；仅在某些题上略少胡算。

**3. 与 gold `A_peer_weighted` 的缺口**  
四模型在本五题中仍主要落在 **`H_equal_mix`、`B_top_anchor`、`D_pure_private`、`E_closest_mimicry`、`F_median_anchor`、`C_uniform_avg`** 之一；共性仍是：**会写社会情绪，难把多名同伴的社会项写成可检验的亲疏加权合成式**。

**4. 相对旧三小的变化**  
纳入 **qwen2.5-14b** 后，七路全错池**缩小**（126 题），Top-5 **题面亦变**（如新增 dining 场景）；**14B** 在题 1 与题 3 的表述略规整，但在题 4、题 5 仍与 Llama 系列同源错误，**不应**因参数量上升即视为已对齐 `A_peer_weighted`。

---

## 附录

- 聚合统计与 `common_wrong_eval_A_universal.csv`：`python3 scripts/analyze_common_wrong_cross_model.py`（当前 **7** 路预测目录参与「全错」判定）。
- 仅保留本文件作为 Top-5 定性材料；**勿**再维护已删除的 `top5_universal_evalA_reasoning.md`。若需用脚本重新生成**不含**下文人工段落的机器稿，可运行 `python3 scripts/export_top5_universal_evalA_reasoning.py`（**会整文件覆盖本 md**，请先 git 备份）；脚本会从 `gpt-4` 的 `eval_A_predictions.jsonl` **写入与各题绑定的完整 `prompt`（题干）**。**本文各模型段落后「简析」及「本地小参数四模型…」为人工撰写**。
