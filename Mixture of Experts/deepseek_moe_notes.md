##### Deepseek MoE paper
- They use **Fine-Grained Expert Segmentation**:
	- They make each expert have less parameters by reducing the FFN hidden dim by a factor of **m**
	- They then increase the number of experts by a factor of **m** and activate **m** times more experts each time.
	- This makes each expert more specialized through greater segmentation of the data
- They also employ **Shared Expert Isolation**:
	- These experts are activated every time, irrespective of the routing network and store shared/basic information to reduce redundancy.
- Note that they use top-2 routing here, where they choose k=2 and then compute soft-gate probabilities weighted between the two chosen experts.
![[Screenshot from 2025-02-14 12-43-04.png]]

##### Deepseek V3 MoE Updates
- The authors develop an auxiliary-loss-free load balancing. This helps to solve the biggest problem with MoE which is an unstable learning process due to this loss.
- They do this through a load balance term added to each routing output affinity score
	- They create a new bias update speed parameter gamma
	- During training, expert utilization is monitored and over-utilized experts will have their bias term decreased by gamma, while less utilized experts will have their bias terms increased by gamma
	- $b_i = b_i - \gamma$
	- $b_i = b_i + \gamma$
- topK outputs:
	- We compute the scores for all experts, where each expert j has a score $s_{i,t}$​.
	- We apply a bias term $b_j$​ to each expert's score.
	- We then **rank** all these modified scores $s_{i,t}$ across all experts j.
	- The **TopK** function selects the **top KrK_rKr​ highest** scores.


















##### Basic Deepseek V2 architecture:
- Hierarchical [[Mixture of Experts (MoE)]] architecture
- The tokens first go through a layer of shared experts, they then go through router experts which are selectively activated using a gating network. Also done in Deepseek R1
- **Global selection**: inputs are routed to an initial pool of experts using softmax affinity scoring
- **Cluster-level pruning**: within each pool, a secondary gating mechanism prunes experts based on entropy constraints (at the start you may want high entropy for exploration, then move to higher entropy for deterministic outputs later on).
- **Final expert assignments**: top-k experts are chosen using either entropy aware gating function, or RL policy for Deepseek R1.

###### Loss updates:
- The authors use three losses:
- **Load balance loss** we saw in switch transformer. Balances the usage of each individual expert.
- **Device balance loss**: the experts are split into groups and assigned to devices. They want all devices to be used relatively equally. 
- **Communication balance loss**: balances the communication load across experts. 

###### Token-dropping strategy:
- They use a capacity factor of 1
- When there are too many tokens sent to one device, the one with the lowest affinity score (routing probability) is dropped
- In training, they also randomly sample 10% of the total tokens and enforce that they can not be dropped. This forces the model to at least see all the tokens.







# Deepseek V3:
- Finds a way to remove the auxiliary losses through a bias term.
- Other engineering improvements in a federated learning setting.










# Deepseek R1
- Introduces RL based expert routing
- Instead of using a learned linear layer with softmax activation for expert routing, Deepseek R1 utilizes a learned RL policy to dynamically assign tokens to experts.
- The policy is as follows

![[DeepseekR1_RL_policy.png]]
- Probability of choosing expert **i** is decided by policy $\pi_{theta}$ for expert $e_i$ given input embedding $u_t$.

- This is trained using [[GRPO]], an update to [[Proximal Policy Optimization (PPO)]] that adds a KL divergence regularization term
	- P(Q): Distribution of token embeddings (states).
	- $\pi_{\theta}$: The current policy network
	- $\pi_{old}$: The old policy network before the update.
	- $A_i$: The advantage function, which estimates how much better a given action (expert selection) is compared to the average action.
	- Clip() clips the fractional output to 1 to prevent drastic policy changes
	- $D_{KL}$ is the KL divergence from the current policy to the reference policy. Not stated what the reference policy is but I assume its a uniform distribution?


- Instead of an auxiliary loss term, they update the bias depending on the usage.
	- If expert i is overused, decrease the bias for expert i by substracting a gamma param. Otherwise, increase $b_i$ by $\gamma$

