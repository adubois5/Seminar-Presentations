### Note:
These notes were copied from obsidian notebook. Therefore, the spacing on the .md file may be weird. Please view the rendered version instead.


### Introduction
- Papers that will be discussed:
	- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
	- Note that there are other system optimizations that could be discussed, but this paper is already 25 pages, and the other content would be too much for one reading group. The most obvious papers that are relevant are: ZeRO-Offload and ZeRO-Infinity
## Why this is important?
- Pretty obvious:
	- If you want to train much larger models, or just run the biggest opensource LLMs: you will need this.
	- If you want to work in industry where the budgets are much larger, computing in a distributed setting is the common practice.

## How does this work?
### Introduce the steps:
- To better understand ZeRO, we need to follow the development cycle of these ideas from start to finish. Therefore, we will quickly go over some core ideas in this field that informed the use of ZeRO-3, before going into the nitty-gritty of how it works.
- Lastly, if you use Pytorch, you have likely been using their Fully Sharded Data Parallel module (FSDP), instead of Deepspeed. From what I have seen, these seem to be conceptually extremely similar, and only really different in implementation, where FSDP is better integrated for Pytorch specific data-structures in the backend.






### Cuda Distributed Functions:
- [[Multi-GPU functions]]
- Talk about the images






### Data Parallelism:
- Each GPU holds the full model, optimizer states, etc.
- Each GPU gets its own batch of data, and therefore runs through training faster
- This requires 3 synchronization points:
	- 1 reduce scatter operation on the data (very small)
	- 1 all reduce on the gradients (depending on the gradient accumulation steps, this could be infrequent)
		- Remember that the most efficient way to all reduce is to run 2 cuda communication steps (reduce-scatter + all-gather)
		- You can also think about it as a Reduce operation followed by a broadcast
- Good compute/communication efficiency, poor memory efficiency







### Pipeline Parallelism:
[[Pipeline parallelism]]
- Split the model into sequential stages, put different stages on different devices; stream micro-batches through the stages so multiple micro-batches are in flight.
- **Pros:** Memory reduction (each device stores only its stage)
	- Lowers peak activation requirement per device.
	- Really good for inference only
- **Cons / Tradeoffs:** pipeline _bubbles_ (idle time) that depend on number of micro-batches *m* and number of pipeline stages *p*. Bubble fraction ≈ (p-1)/(m+p-1)
	- There is a ramp-up time when you start training since mini-batches are usually grouped and sent through a queue like system
##### Note for me:
- Each activation is sent forward in the pipeline and used as input for the next layer
- In backwards, we send gradients backward ($\delta L / \delta \left(activations\right)$)

### Model Parallelism:
[[Tensor Parallelism (TP)]] (Megatron-LM paper)
- Model Parallelism (aka Tensor Parallelism) involves splitting the model's tensors across GPUs. An example of this is in the Megatron-LM paper that demonstrates a method for splitting linear and transformer layers across GPUs.



##### Images of layer splitting
![[Screenshot from 2025-07-29 12-17-35.png|500]]
- Linear layers requires 2 points of synchronization:
	- 1 at f in the backward pass
	- 1 at g in the forward pass
- Self-Attention also only needs the same 2 points of synchronization


#### Vertical Splitting

| Normal Linear Layer                                                                                                                                                                                                                                                                      | Column-wise MP Linear Layer:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $X = \begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix}$<br>$A = \begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix}$<br>$Y = ReLU\left(\begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix} \begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix}\right)$       <br>$= ReLU\left(\begin{bmatrix}7 & 10 \\1 & 2\end{bmatrix}\right)$ | $X =\begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix}$<br>$\begin{bmatrix}A_1, A_2\end{bmatrix}  = \begin{bmatrix}\begin{bmatrix}1 \\3\end{bmatrix}\begin{bmatrix}2\\4\end{bmatrix}\end{bmatrix}$<br><br>$\begin{bmatrix}Y_1, Y_2 \end{bmatrix}$ = $\begin{bmatrix}ReLU\left(\begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix}\begin{bmatrix}1 \\ 3\end{bmatrix}\right), ReLU\left(\begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix}\begin{bmatrix}2 \\ 4\end{bmatrix}\right)\end{bmatrix}$<br>= $\begin{bmatrix}ReLU\left(\begin{bmatrix}7 \\ 1\end{bmatrix}\right), ReLU\left(\begin{bmatrix}10 \\ 2\end{bmatrix}\right)\end{bmatrix}$ |


#### Horizontal Splitting:

| Normal Linear Layer                                                                                                                                                                                                                                                                                       | Horizontal-wise MP Linear Layer:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $X = \begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix}$<br>$A = \begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix}$<br>$Y = ReLU\left(\begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix} \begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix}\right)$                        <br>$= ReLU\left(\begin{bmatrix}7 & 10 \\1 & 2\end{bmatrix}\right)$ | $X =\begin{bmatrix}1 & 2 \\1 & 0\end{bmatrix}$ $\rightarrow$ $X_1 = \begin{bmatrix}1 \\ 1\end{bmatrix}$, $X_2 = \begin{bmatrix}2\\0\end{bmatrix}$<br>$\begin{bmatrix}A_1, A_2\end{bmatrix}  = \begin{bmatrix}\begin{bmatrix}1, 2\end{bmatrix}\\\begin{bmatrix}3, 4\end{bmatrix}\end{bmatrix}$<br><br>$\begin{bmatrix}Y_1, Y_2 \end{bmatrix}$ = $\begin{bmatrix}ReLU\left(\begin{bmatrix}1 \\ 1\end{bmatrix}\begin{bmatrix}1, 2\end{bmatrix} + \begin{bmatrix}2 \\ 0\end{bmatrix}\begin{bmatrix}3,  4\end{bmatrix}\right)\end{bmatrix}$    <br>= $\begin{bmatrix}ReLU\left(\begin{bmatrix}1, 2 \\ 1, 2\end{bmatrix} + \begin{bmatrix}6, 8 \\ 0, 0\end{bmatrix}\right)\end{bmatrix}$ |
|                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

- You need to split the data vertically for computation:
	- If you split the rows you end up with: Y = GeLU($X_1 A_1 + X_2 A_2$)
	- Since GeLU is a non-linear function, GeLU($X_1 A_1 + X_2 A_2$) =/= GeLU($X_1 A_1$) + GeLU($X_2 A_2$) 








![[Screenshot from 2025-07-29 12-22-03.png|400]]
- A transformer block needs 4 points of synchronization:
	- 2 for Self attention
	- 2 for FFN





- Poor compute/communication efficiency, good memory efficiency. 
- MP also reduces what is called "the granularity of computation". Granularity refers to the amount of computation done per unit of communication. So coarse grain parallelism: each device does a lot of computation before needing to communicate with other devices. This is good.

- MP has significant costs especially when working between nodes. Since we are splitting layers vertically (each layer itself is split between GPUs), we incur large communication costs when running a layer. These costs can be mitigated through hardware advances such as NV-link for a pair of GPUs or NV-Switch for a 8 GPU node, but communicating through PCIe when moving between nodes is too expensive and inefficient.
	- **We tested a 40B parameter model using Megatron-LM across two DGX-2 nodes and observe about 5 T f lops per V100 GPU (less than 5% of hardware peak)**





### CPU-Offloading
- **Idea:** exploit larger but slower CPU / NVMe memory to hold parts of model state that are not currently in use
- Then you can prefetch to the data GPU before use, and overlap transfers when possible.
- **Issue** "Up to 50% of training time can be spent on GPU-CPU-GPU transfers". ZeRO-3 reduces memory consumption significantly **without storing the model states to CPU memory**, whose bandwidth is severely constrained due to PCI-E.





### Motivation Recap:
- To fully understand where we can eek out a bit more memory gain, we must first understand: where is all our memory going when training ML models?
- We split the memory into two sets:
	- Model states: optimizer states (momentum and variances in Adam), gradients, and parameters
	- Residual states: activation, temporary buffers, unusable fragmented memory
- Therefore, the main contribution of this paper is ZeRO: Zero Redundancy Optimizer.
- ZeRO's core idea: don't replicate the model across DP ranks -- partition it. The goal is to have the memory efficiency of MP, with the ease of DP








### Optimizing Model State Memory:
- The previous approaches require the model states to be kept over the entire training process, even though not all model states are required all the time during training
- Therefore, they define ZeRO-DP which combines DP with MP:
	- Removes the memory state redundancies across DP processes by partitioning model states instead of replicating them
	- Has a dynamic communication schedule for training
- We will go through these advances in 3 steps:
	1. Optimizer State Partitioning ($P_{os}$): 4x memory reduction, same communication volume as DP
	2. Add Gradient Partitioning ($P_{os + g}$): 8x memory reduction, same communication volume as DP
	3. Add Parameter Partitioning ($P_{os+g+p}$): Memory reduction is linear with DP degree $N_d$. 50% increase in communication volume (they say it is a minor increase).
- Talk about FSDP experiments and importance of compute and communication tradeoffs?






#### Partitioning Visual (Only model states):
![[Screenshot from 2025-09-19 12-17-35.png]]
- $\Psi$ (PSI) denotes model size (number of parameters)
- K denotes the memory multiplier of optimizer states
- $N_d$ denotes DP degree.
- Assume a model size of 
	- Ψ = 7.5B
	- DP of Nd = 64
	- K = 12 based on mixed-precision training with Adam optimizer

- Note that ZeRO will only improve the persistent memory footprint. With this method, the largest layer's full parameters must fit in GPU memory at compute time.


#### Mixed Precision Training:
- In mixed precision training, your forward and backward passes are computed in fp16 (weights and activations).
- However, to effectively compute and apply the updates after backward propagation, we need to use a fp32 copies to maintain stable optimization.
- You need 
	- fp16 copy of parameters used in computation (2$\Psi$)
	- fp16 of the gradients ($2\Psi$)
	- fp32 copy of the parameters for stability ($4\Psi$)
	- fp32 copy of the momentum ($4\Psi$)
	- fp32 copy of variances ($4\Psi$)
- This is K = 12 for Adam related parameters, + 4 for forward computation. Total memory requirement is 16$\times$ number of parameters (Billions)

###### Note: why even do mixed precision training:
- Activation is very expensive.
- For a transformer based implementation, the activation memory is proportional to:  transformer layers ×
hidden dimensions × sequence length × batch size.
- For GPT2 like models that is about 12 × hidden dim × batch × seq length × transformer layers
- This is very expensive for moderate models

- Once you get to HUGE models, the extra memory costs in just doing mixed precision training become too big. (1 trillion $\times$ 12 etc.)





### Optimizing Residual State Memory (ZeRO-R)
#### Partitioned activation checkpointing (Pa):
- partition activations among devices so they’re not fully replicated (reduces activation memory proportional to MP degree). ZeRO can offload activation partitions to CPU if needed. [arXiv](https://arxiv.org/pdf/1910.02054)

#### Constant-size temporary buffers: 
- Fixed size buffers are pre-loaded during training to avoid temporary buffers from blowing up as model size increases.
- This reduces efficiency slightly by needing more synchronization points (smaller buffers), but prevents OOM errors.




#### Memory de-fragmentation:
- It is possible to run out of usable memory even when there is plenty of available memory. This can happen with memory fragmentation. A request for a memory will fail if there isn’t enough contiguous memory to satisfy it, even if the total available memory is larger than requested
- Memory fragmentation is a result of interleaving between short lived and long lived memory objects. 
- During the forward propagation activation checkpoints are long lived but the activations that recomputed are short lived.
- Similarly, the backward computation, the activation gradients are short lived while the parameter gradients are long lived. 
- ZeRO performs memory defragmentation by moving activation checkpoints and gradients to pre-allocated contiguous memory buffers.





### Speedup from ZeRO:
![[zero-speedup.png]]
- Models with greater than 40B parameters requires MP across nodes to train.

### ZeRO + MP
- ZeRO does not use MP, but they are compatible. It allows you to further reduce the memory footprint by a factor of $N_d$, but increases communication costs, and set-up/coding costs






## Insights & Overview/Recap
### Zero-DP
- ZeRO-3 greatly reduces temporary memory. At the forward pass, an all gather is still performed within a DP group and the full layer needs to fit onto the GPU.
- Then, once the activations are computed, they are reduce-scattered across the data-parallel GPUs so that persistent memory footprint is minimized.
### Zero-R:
- ZeRO removes the memory redundancies in MP by partitioning the activations checkpoints across GPUs, and uses allgather to reconstruct them on demand. The activation memory footprint is reduced proportional to the MP degree. Also uses constant size activation buffers for very large models, and pre-allocates fixed buffers for short-term temporary allocations. 





## Communication Analysis:
### DP (Baseline)
-This requires two synchronization points:
	- 1 reduce scatter operation on the data.
	- 1 all reduce on the gradients (depending on the gradient accumulation steps, this could be infrequent)
- This operation results in a total communication volume of **$2\Psi$** per training step




### $P_{os+g}$: Optimizer State & Gradient Partitioning
- Here, full model is still replicated on each device.
- Only optimizer states and gradients are partitioned after calculation (still a significant memory gain)
- When $P_{os}$ and $P_g$ are used ($P_{os+g}$), ZeRO replaces the standard gradient all-reduce with a **scatter-reduce** operation. 
- An **all-gather** operation is then performed to collect all updated parameters after local updates (volume). The total communication volume is $\Psi + \Psi = \mathbf{2\Psi}$, which is **exactly the same as the baseline DP**





### $P_{p}$: Parameter Partitioning
- With $P_{os+g+p}$, parameters are partitioned. Parameters required by a device outside its partition during forward/backward propagation are received via broadcast.
- This involves rescheduling the parameter all-gather across the entire forward propagation (volume $\Psi$), followed by discarding the parameters. This must happen again for backward propagation (volume $\Psi$). 
- Combining these two all-gathers ($2\Psi$) with the gradient reduce-scatter ($\Psi$) yields a total communication volume of **$3\Psi$**, which is **1.5x compared to the baseline DP**






### $P_{a}$: Partitioned Activation Checkpointing
- $P_a$ requires an additional all-gather operation (volume $seq\ length \times hidden\ dim$) before the forward recomputation during back-propagation, typically one per transformer block.
- The communication overhead of $P_a$ is less than **10% of the original communication volume** incurred by baseline MP (like Megatron-LM). 
- Crucially, $P_a$ can reduce the activation memory by the MP degree, allowing a proportional increase in batch size. Since DP communication volume is inversely proportional to batch size, this can result in an **order-of-magnitude decrease in data-parallel communication volume**, significantly boosting efficiency when DP communication is a bottleneck
