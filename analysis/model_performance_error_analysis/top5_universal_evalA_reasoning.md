# eval_A：六模型全错 Top 5 题（错选主导规则尽量分散）
选取方式：在 `common_wrong_eval_A_universal.csv` 中按 `task_id` 排序，贪心选取 **5** 道题，使各题 6 人错选直方图的 **众数规则** 尽量不重复；不足时用后续 `task_id` 补齐。
每题 gold 均为 `A_peer_weighted`。`Reasoning` 取自 `raw_response` 中 `Reasoning:` 至 `Choice:` 段（无则取全文前缀）。

---

## 1. `evalA__d1_cele_02__alpha_high__disp_high__skew_low__00`

- **scene_id**: `d1_cele_02` | **cell_id**: `alpha_high__disp_high__skew_low` | **gold_rule_id**: `A_peer_weighted`
- **6 人错选直方图** (`parsed_rule_id`): `{"C_uniform_avg": 1, "B_top_anchor": 4, "E_closest_mimicry": 1}`
- **众数错选规则**: `B_top_anchor`

### DeepSeek-R1-671B

**parsed_rule_id**: `C_uniform_avg`

First, I need to understand the scenario. The protagonist has a budget of $3,486 for a simple backyard birthday party. But after seeing what others are spending, especially Maya (a close friend), who's spending $4,924 on an extravagant party, they're reconsidering. Other parents are spending around the protagonist's budget: Ben at $3,112, Chloe at $3,504, Dave at $3,561. Maya is the standout with a much higher spend.

The question is whether to upgrade to match Maya or stick to the original plan. I need to pick the reasoning rule that best explains what the protagonist would commit to.

Now, looking at the candidate rules:

- (A) Build on the baseline with a social contribution where close ties weigh heavier. This lands near $7,441.

- (B) Take the unweighted mean of what parents are doing and stack that above the private floor. Arrives near $6,789.

- (C) Route the peer component entirely through the most relationally close parent and layer it on top of the personal floor. Around $7,793.

- (D) Replace the private baseline with the single most intimate parent's move and stop there. Ends at $4,924.

I need to reason based on the social dynamics. The protagonist mentions Maya as "my college roommate turned mom-confidante," which indicates a very close relationship. In contrast, Ben is "that quiet dad from the pediatrician’s Slack," and Dave is "someone I only recognize from daycare pickup," so they're acquaintances, not close ties.

The protagonist is influenced by Maya's extravagance and feels pressure from the group chat. The key point is the relational closeness to Maya.

Rule (D) says: replace the private baseline with the single most intimate parent's move and stop there. That would mean adopting Maya's $4,924 budget exactly. That seems

…[truncated for file length]…

### Qwen

**parsed_rule_id**: `B_top_anchor`

Okay, let's tackle this. The protagonist has a budget of $3,486 for their child's birthday party. They're in a group chat where other parents are discussing their plans. Maya, a close friend, is spending $4,924 on a lavish party. Others like Ben, Chloe, and Dave have expenses around $3,112 to $3,561, which are closer to the protagonist's original budget.

The question is whether they should upgrade to match Maya's spending or stick to their plan. The candidate rules suggest different ways to adjust the budget based on social influences. Let's look at each option.

Option A suggests adding a social contribution where close ties matter more. If Maya's impact is significant, maybe this pushes the budget up. But how much? If the average of others' spending is around $3,500, adding that to the original might not reach $7,441, but maybe with weighting, it could. Not sure.

Option B takes the mean of all the mentioned spends: $4,924 (Maya) + $3,112 (Ben) + $3,504 (Chloe) + $3,561 (Dave) = total $15,101 divided by 4 equals $3,775.25. Then stack a social term above the private floor. Wait, the description says "unweighted mean" and "stack that social term above the private floor." So maybe it's taking the average of all the mentioned parents' expenditures and adding that to the private floor? That would be $3,486 + $3,775 = $7,261, which is near $6,789? Hmm, maybe not. The example output says around $6,789. Maybe the mean is calculated differently? Or perhaps the rule is to take the mean as the social component and add it to the private floor. Alternatively, maybe the mean is around $3,303 (if considering others closer to protagonist's number), but the answer says around $6,789. Maybe the average is $3,303, added to the original, 3,486 + 3,303 = ~6,

…[truncated for file length]…

### gpt-4

**parsed_rule_id**: `E_closest_mimicry`

The protagonist values their close relationship with Maya, who is their college roommate turned mom-confidante, and Maya's party plans are clearly extravagant. Choosing the rule that mirrors Maya's plans directly aligns with the protagonist’s relational closeness to her. Thus, adopting Maya's exact budget reflects the importance of this intimate relationship.

### ollama-llama3-1-latest

**parsed_rule_id**: `B_top_anchor`

The protagonist is deciding whether to upgrade their party budget to match Maya's extravagance. However, the story highlights that Maya is a college roommate and mom-confidante, indicating a close tie, but also emphasizes the protagonist's own quiet plan and concern about matching someone else's extravagant expenses. 

Given this context, the reasoning rule that best explains the protagonist's decision would be to take their private baseline ($3,486) as the foundation and adjust slightly upwards to match the level of intensity or extravagance shown by Maya, but not exceeding it.

Choice: C

### ollama-llama3-8b

**parsed_rule_id**: `B_top_anchor`

The protagonist wants to fit in with the group and doesn't want to be seen as cheap, but also values their own quiet plan. They should choose a compromise that balances these desires.

### ollama-qwen2-7b

**parsed_rule_id**: `B_top_anchor`

The protagonist should consider upgrading their plan to match the cost of their closest friend's party (Maya's), as close ties are given more weight in this scenario.


---

## 2. `evalA__d1_cele_02__alpha_high__disp_high__skew_mid__01`

- **scene_id**: `d1_cele_02` | **cell_id**: `alpha_high__disp_high__skew_mid` | **gold_rule_id**: `A_peer_weighted`
- **6 人错选直方图** (`parsed_rule_id`): `{"C_uniform_avg": 1, "E_closest_mimicry": 5}`
- **众数错选规则**: `E_closest_mimicry`

### DeepSeek-R1-671B

**parsed_rule_id**: `C_uniform_avg`

First, I need to understand the scenario. The protagonist is planning a birthday party for Leo and is influenced by what other parents in the group are spending. Privately, they want to spend around $803 for a simple party. But friends and acquaintances like Sarah (best friend), Ben, and Lena have spent much more: $1,589, $1,478, and $1,178 respectively. The protagonist is torn between sticking to their private plan or matching the others' spending.

Now, I have to choose the best reasoning rule from the options that explains what the protagonist would commit to. The options are:

(A) Stay on the private floor: just the personal need and practical cost, about $803.

(B) Scale the budget by how close each parent feels, adding social pull to the private floor, resulting in around $2,207.

(C) Copy whatever the single closest parent is doing outright, ignoring private floor, about $1,589.

(D) Give every parent an identical share of the peer aggregate, then add that to the baseline, resulting in around $2,095.

I need to reason briefly in 1-3 sentences and then pick one letter.

Key elements from the scenario:

- The protagonist has a private preference for simplicity and low cost ($803).

- Social influence is strong, especially from Sarah, who is described as "best friend" and "practically co-parents," indicating high closeness.

- Other parents like Ben and Lena are less close; Ben is "only recognize from the welcome Slack," and Lena is "whose feed I scroll but never chat with," so less intimate.

- The protagonist feels pressure from seeing their spending and social media posts, making it seem effortless.

What would the protagonist do? The scenario says: "Do I match their sparkle or stick to my quiet plan?" This shows internal conflict.



…[truncated for file length]…

### Qwen

**parsed_rule_id**: `E_closest_mimicry`

Okay, let's see. The scenario is about deciding a budget for a kid's birthday party in a neighborhood group. The main character originally thought of a simple backyard setup costing around $803. But then friends in the group have spent more: Sarah spent $1,589, Ben $1,478, Lena $1,178. Now they're trying to decide whether to stick to their original plan or match the others.

The question is to choose the best rule that explains what the protagonist would commit to. The options are A through D. 

Looking at the rules:

A says to stick with the personal practical cost, $803. That's the original plan. But the protagonist is influenced by others' spending, so maybe not A.

B suggests scaling based on closeness in daily life. The protagonist mentions Sarah is their best friend (so higher influence), Ben is only recognized from Slack (maybe less close), Lena's feed they scroll but don't chat (maybe moderate). So maybe B adds the social pull, leading to a higher total than A. 

C says to copy the closest parent entirely, which would be Sarah's $1,589. But the problem is that the protagonist might feel peer pressure but not sure if they'd fully copy. The question says "the ONE rule that best explains what the protagonist would commit to". The initial thought was to stick with $803 but influenced by others. 

D is taking an average of the peers and adding to the baseline. The numbers given for D are around $2095, which is higher than all the examples. But the problem is that the protagonist might not just average it but perhaps be influenced more by the closest friend. 

The protagonist's best friend Sarah is a big influence. The text says "co-parents with us" so Sarah is very close. Ben is less close (only known from Slack), Lena is somewhere in be

…[truncated for file length]…

### gpt-4

**parsed_rule_id**: `E_closest_mimicry`

The protagonist is influenced by both personal preferences and social pressure, but the strongest influence comes from the closest relationship, which is with Sarah. Her budget notably impacts the protagonist's consideration due to their closeness. Therefore, they are most likely to match Sarah's party expense exactly.

### ollama-llama3-1-latest

**parsed_rule_id**: `E_closest_mimicry`

The protagonist is being influenced by their desire to fit in and match the sparkle of others, as evident from their consideration of the neighbors' elaborate party budgets. However, they also have a private preference for a simpler backyard celebration. The most fitting rule is (C) Copy whatever the single closest parent is doing outright, which aligns with the protagonist's aim to emulate Sarah, who is their best friend and practically co-parents.

Choice: C

### ollama-llama3-8b

**parsed_rule_id**: `E_closest_mimicry`

The protagonist is torn between sticking to their original simple plan and keeping up with the extravagant displays on Instagram. They seem to be intimidated by their peers' bold spending habits.

### ollama-qwen2-7b

**parsed_rule_id**: `E_closest_mimicry`

The protagonist is influenced by the public displays of party spending from close friends and acquaintances in their community. This external pressure encourages them to consider matching higher budgets for a comparable level of perceived prestige.


---

## 3. `evalA__d1_cele_02__alpha_high__disp_low__skew_low__01`

- **scene_id**: `d1_cele_02` | **cell_id**: `alpha_high__disp_low__skew_low` | **gold_rule_id**: `A_peer_weighted`
- **6 人错选直方图** (`parsed_rule_id`): `{"G_counter_conformist": 2, "H_equal_mix": 1, "D_pure_private": 3}`
- **众数错选规则**: `D_pure_private`

### DeepSeek-R1-671B

**parsed_rule_id**: `G_counter_conformist`

First, I need to understand the scenario. The protagonist is in a young-parent group chat where everyone is sharing their first-birthday party budgets. The people mentioned are:

- Ben, a neighbor: spent $2,272

- Priya, carpool partner for daycare: spent $2,685

- Maya, from baby yoga (vaguely known): spent $2,358

- Alex, twin's spouse (close family): spent $2,666

The protagonist's own initial plan is $2,172 for a simple party.

Now, the chat expects the protagonist to reveal their budget, and they're feeling pressure from the others' extravagant spending.

I have to choose one rule that best explains what the protagonist would commit to. The options are:

(A) Invert the peer signal: the more peers spend, the lower the protagonist goes, even to near $0.

(B) Elevate the baseline by a weighted peer aggregate where closer bonds matter more, leading to around $4,496.

(C) Half private floor, half closeness-weighted peer aggregate, no modulation, around $2,356.

(D) Anchor entirely on the private baseline, ignoring peers, so $2,172.

I need to think about what the protagonist would do based on their feelings. The scenario says: "My thumb hovers over the keyboard, digits burning unsent." This suggests hesitation and possibly not wanting to conform to the high spending.

Also, the protagonist describes their plan as "quiet," "simple," "no frills, no witnesses," implying a preference for modesty and perhaps discomfort with the extravagance.

Now, looking at the relationships:

- Ben: neighbor, always practical – not very close, but practical.

- Priya: carpool for daycare – closer, daily interaction.

- Maya: vaguely known from baby yoga – not close.

- Alex: twin's spouse – very close, family.

In rule (B), it's about weighting closer bonds mo

…[truncated for file length]…

### Qwen

**parsed_rule_id**: `G_counter_conformist`

Okay, let's see. The user is in this parent group chat where everyone is sharing their birthday party budgets, which are all over $2,000. The protagonist's own plan is $2,172, but now they're under pressure to reveal their budget. The options are A to D.

Hmm, the protagonist's original plan is $2,172, which is already a bit under some of the others. The question is how they'd react to the peer pressure. Let's break down the rules.

Rule A says to invert the peer signal, going even lower. But the protagonist is already below some. But maybe if peers are going higher, invert by going lower. However, their plan is already under most except Ben's $2,272. So maybe not all the way to zero.

Rule B suggests elevating the baseline using a weighted average, giving more weight to closer connections. The problem here is that some people mentioned are closer (like Priya, the neighbor Ben, Alex) while Maya is a vague connection. This might pull the budget higher, maybe to $4k? But that seems way over. But the number given is $4,496. But the protagonist's own plan is $2k. If the weighted peers are higher, maybe B would push it higher. But the options don't align with the protagonist's current number.

Rule C is a balance: half private floor (so $2,172) and half average weighted by closeness. The answer here would be averaging $2,172 and the peer aggregate. The peer numbers: Ben is $2,272, Priya $2,685, Maya $2,358, Alex $2,666. Weighting closer relationships more. So maybe the average of the closer ones (Priya, Ben, Alex) would be higher. Let's see: Priya's $2685, Ben's $2272, Alex $2666. Average of those three is (2685 + 2272 + 2666)/3 = 7623/3 = 2541. Then maybe include Maya with lower weight. But C is half private floor (2172/2=1086) plus half of the

…[truncated for file length]…

### gpt-4

**parsed_rule_id**: `H_equal_mix`

Given the strong social pressure to reveal a party budget, coupled with the protagonist's awareness of various budgets shared within the group, they are likely to adopt a strategy that considers peer influence but also retains some personal baseline due to the practical nature of their initial plan. Rule C balances these influences, producing a compromise that acknowledges both personal preference and peer comparison.

### ollama-llama3-1-latest

**parsed_rule_id**: `D_pure_private`

The protagonist is hesitant and seems uncertain about how to determine the party budget, as evidenced by their thumb hovering over the keyboard. They appear to be torn between revealing a higher budget that matches their neighbors' expectations and sticking with their original, more modest plan.

### ollama-llama3-8b

**parsed_rule_id**: `D_pure_private`

The protagonist's thumb hovers over the keyboard, suggesting hesitation and potential influence from the peer group chat. However, the protagonist still prioritizes their private baseline, indicating a desire to align with their personal preferences rather than conform to the group's expectations.

### ollama-qwen2-7b

**parsed_rule_id**: `D_pure_private`

The protagonist's personal budget has been set independently and is based on her own considerations of cost and celebration style. She has observed her peers' spending but does not intend to adjust her budget based on their choices. Her decision-making process focuses on her private preferences and financial limits rather than the escalating costs suggested by her neighbors' celebrations.


---

## 4. `evalA__d1_cele_02__alpha_high__disp_low__skew_low__03`

- **scene_id**: `d1_cele_02` | **cell_id**: `alpha_high__disp_low__skew_low` | **gold_rule_id**: `A_peer_weighted`
- **6 人错选直方图** (`parsed_rule_id`): `{"F_median_anchor": 2, "D_pure_private": 1, "H_equal_mix": 3}`
- **众数错选规则**: `H_equal_mix`

### DeepSeek-R1-671B

**parsed_rule_id**: `F_median_anchor`

First, I need to understand the scenario. The protagonist is planning a baby's first birthday party and is influenced by what other parents in their group are spending. They have a private baseline budget of $926, but they see others spending more: Tamara at $1,051, Ben at $1,371, and Lena at $1,030. The chat is buzzing with excitement, making the protagonist feel pressured to spend more.

Now, I have to choose the best reasoning rule from the options that explains what the protagonist would commit to. The options are:

(A) Use the median of what the parents are doing as the social pull above the private baseline — around $1,851.

(B) Mix the private floor and the weighted social pull at a flat 1:1 ratio, paying no attention to the individual's peer-sensitivity — near $1,044.

(C) Stick with what the private baseline dictates without any lift from what the circle is doing — the party budget sits at $926.

(D) Use a proximity-graded blend of what the parents are doing as a lift above the personal need — the party budget sits near $1,949.

I need to calculate the values mentioned to verify them and see which one fits.

First, list the spending mentioned:

- Tamara: $1,051

- Ben: $1,371

- Lena: $1,030

Also, the protagonist's baseline is $926.

Now, for (A): Median of what parents are doing. The parents' spending: $1,051, $1,371, $1,030. To find the median, order them: $1,030, $1,051, $1,371. Median is the middle one: $1,051. But option says "around $1,851" – that doesn't match. $1,851 is way higher. Perhaps it's a mistake? Or maybe it includes something else. The option says "median of what the parents are doing" and "around $1,851". But based on the numbers, median should be around $1,051. $1,851 is close to Ben's $1,371? No, that's not me

…[truncated for file length]…

### Qwen

**parsed_rule_id**: `D_pure_private`

Okay, let's try to figure this out. The user is trying to decide their baby's first birthday party budget based on others in their social circle. The original budget was $926, but others have spent more. Tamara spent $1,051, Ben $1,371, Lena $1,030. The user is feeling pressure to spend more.

Looking at the candidate rules: 

A says use the median of others, which would be around $1,851? Wait, the given numbers are Tamara 1051, Ben 1371, Lena 1030. If we sort those: 1030, 1051, 1371. The median is 1051. But option A says median would be around 1851, which doesn't make sense. Maybe that's a mistake? Or maybe the median of more people? Hmm, maybe the problem assumes others have higher values not listed? Not sure. Wait, the scenario mentions "parents' group chat" and the examples given are Tamara, Ben, Lena. Maybe the median in the group is higher?

Alternatively, maybe option A and D's numbers are calculated differently. Let's see. The options are about how to combine the social influences and personal budget. The user is torn between their original plan ($926) and the glittering expectations (higher amounts). 

Option C is sticking with personal budget. Option B mixes private and social at 1:1 ratio. The social pull would be an average of others? Let's see: Tamara's 1051, Ben's 1371, Lena's 1030. Average is (1051 + 1371 +1030)/3 = 3452/3 ≈ 1150.67. If B is average of private floor (926) and the social pull (maybe that average 1150?), then (926 + 1150)/2 = 1038, which is near 1,044 as in B. So that might be right. 

Option D uses a proximity-graded blend, which might mean that closer friends have more weight? Lena is a close friend, so maybe her $1030 and Tamara, but Ben is just a jogging buddy. If proximity-graded, maybe Lena and Tamara (wh

…[truncated for file length]…

### gpt-4

**parsed_rule_id**: `H_equal_mix`

The protagonist seems influenced by the peer pressure from close friends and acquaintances, making option B the most realistic. It balances the original budget with the surrounding expectations, considering both personal plans and social cues from their immediate circle.

### ollama-llama3-1-latest

**parsed_rule_id**: `H_equal_mix`

The protagonist is torn between their original budget and the expectations of others, suggesting they are influenced by social pressures. However, they also mention having a "quiet plan" in notes, indicating that they have a personal baseline for the party's cost.

### ollama-llama3-8b

**parsed_rule_id**: `H_equal_mix`

The protagonist is torn between their original plan and the influence of others' party budgets. They seem to be considering the social expectations and peer pressure from Tamara, Ben, and Lena.

### ollama-qwen2-7b

**parsed_rule_id**: `F_median_anchor`

The protagonist is experiencing significant social pressure from their peers' plans, which are influencing her decision-making process. She acknowledges the varying costs proposed by Tamara, Ben, and Lena, reflecting a concern for maintaining community standards while also staying true to her initial budget.


---

## 5. `evalA__d1_cele_02__alpha_high__disp_low__skew_mid__00`

- **scene_id**: `d1_cele_02` | **cell_id**: `alpha_high__disp_low__skew_mid` | **gold_rule_id**: `A_peer_weighted`
- **6 人错选直方图** (`parsed_rule_id`): `{"C_uniform_avg": 3, "B_top_anchor": 2, "F_median_anchor": 1}`
- **众数错选规则**: `C_uniform_avg`

### DeepSeek-R1-671B

**parsed_rule_id**: `C_uniform_avg`

First, I need to understand the scenario. Maya is deciding her party budget based on what others in her young-parent group chat are sharing. She has a private plan of $2,321 for a simple gathering, but she's seeing others spending more: Liam at $2,762, Chloe at $3,292, Noah at $3,279, Emma at $2,629, and Sophia at $6,193. As she's about to share her budget, she hesitates between sticking to her plan or matching the group's expectations.

Now, I have to choose the best reasoning rule from the four options that explains what Maya would commit to. The options are:

(A) Let each of the parents tug the budget based on relationship weight, layered above baseline, near $5,749.

(B) Treat each parent as one vote in a democratic average above baseline, near $5,269.

(C) Let the middle parent set the comparison, layered on private floor, near $4,984.

(D) Pin to whatever the nearest parent is doing, stacked on baseline, near $7,350.

I need to figure out which rule Maya is likely to follow based on her social interactions.

Looking at her relationships:

- Liam: She occasionally likes his feed, so not very close.

- Chloe: She often swaps daycare tips, so closer, like a neighbor or frequent contact.

- Noah: She barely knows him, so distant.

- Emma: Another occasional poster, so not very close.

- Sophia: Her go-to for stroller advice, so very close and trusted.

Maya seems to have varying levels of closeness with these parents. Sophia is highlighted as her "go-to," indicating a strong relationship. Chloe is mentioned as a neighbor with whom she swaps tips, so also close. Others are less significant.

The rules involve "layered above the baseline" or similar phrases. The baseline might be her private plan of $2,321.

Now, let's interpret the rules:


…[truncated for file length]…

### Qwen

**parsed_rule_id**: `B_top_anchor`

Okay, let's see. Maya is in a parent group chat where everyone is sharing their baby's first birthday party budgets. The other parents have spent increasing amounts, from around $2,700 up to $6,193. Maya originally thought of a simpler party with a lower budget. Now she's considering whether to stick to her plan or go along with what others are doing.

The question is asking which reasoning rule best explains what Maya would commit to. The options are A, B, C, or D. Each rule seems to involve combining the baseline need with something related to the other parents' spending.

Looking at the candidate rules:

A) Proportional to the weight of relationships: So if some parents are close to her, their spending might influence her more. For example, Sophia (stroller advice) spent $6k, which is the highest. If Maya values closer relationships more, this might pull her budget higher. The result here is near $5,749.

B) Democratic vote, flat average: Take all the amounts and average them. Let's calculate that. The numbers from the scenario: Liam $2762, Chloe $3292, Noah $3279, Emma $2629, Sophia $6193. The average of these (assuming all 5 are included) would be (2762 + 3292 + 3279 + 2629 + 6193)/5. Let me add them up: 2762+3292=6054, +3279=9333, +2629=11962, +6193=18155. Divided by 5: 18155/5=3631. But the options mention the average being near $5,269, which doesn't match. Wait, maybe there's more data here. Wait the scenario mentions Maya's own budget as $2321. But the other parents (excluding Maya) are Liam, Chloe, Noah, Emma, and Sophia. Wait the amounts are $2762, $3292, $3279, $2629, $6193. So total is 2762 + 3292 = 6054; 6054 + 3279 = 9333; +2629=11962; +6193=18155. Average is 18155/5=3631. But the options B says "flat average above the baseli

…[truncated for file length]…

### gpt-4

**parsed_rule_id**: `F_median_anchor`

Maya seems influenced by various parents, but her budget consideration likely reflects a balance between her initial plan and the practical realities of the group. The middle-of-the-road reasoning acknowledges her desire to conform somewhat without going to extremes, making option C the most fitting for her situation.

### ollama-llama3-1-latest

**parsed_rule_id**: `C_uniform_avg`

Maya considers herself just one of many parents in the group chat, rather than someone who needs to match or outdo the others. She's not trying to establish a reputation or hierarchy within the group.

### ollama-llama3-8b

**parsed_rule_id**: `C_uniform_avg`

Maya is torn between her original simple plan and the pressure from others to match their extravagant celebrations. She seems to be influenced by the group's expectations, but wants to remain true to herself.

### ollama-qwen2-7b

**parsed_rule_id**: `B_top_anchor`

Maya was influenced by the extravagant budgets shared in the group chat, feeling pressured to match or exceed these amounts given their visibility and the perceived expectations of her peers.

