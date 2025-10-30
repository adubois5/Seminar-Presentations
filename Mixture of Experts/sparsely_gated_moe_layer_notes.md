- Capacity of an NN to absorb information is limited by the number of parameters.
	- We'll actually see in future slides that this relationship is actually an exponential decay function with lower test loss as number of parameters increases.
	- So, conditional computation where only parts of the model are active at once allows for greater parameters without increasing computation reqs.
This paper uses [[LSTMs]] as their LM (2017 release before transformers)





##### Their Approach:
- The MoE layer consists of n experts, each a basic [[Feed Forward Network]], and a trainable gating network which selects a sparse combination of experts to process each input. All parts of the model are trained jointly through backprop.
- They apply MoE convolutionally between stacked LSTM layers as seen below. The [[Mixture of Experts (MoE)]] is called once for every poistion in the text.
##### Diagram of the MoE Layer
![[GatedMoELayer.png]]
##### Structure of MoE:
- The Mixture-of-Experts (MoE) layer consists of a set of n “expert networks" E1, · · · , En, and a “gating network" G whose output is a sparse n-dimensional vector.
- We only require that the experts accept the same sized inputs and produce the same-sized outputs. Common practice is that all experts have the exact same architecture.
- The architecture shown above is a "flat MoE" (standard). However, if you want to define many experts (up to 1000s) you can reduce the branching factor by using a hierarchy of gating networks.
	- Deepseek uses a version of Hierarchical MoE, where each gating network is a Reinforcement learning agent.
##### Description of the diagram
- The gating scores $G(x)_2$ and $G(x)_{n-1}$ act as multiplicative weights for the outputs of the selected experts.








##### Challenges with MoE models at the time
- Branching is expensive GPUs are faster at arithmetic compared to branching (not solved here)
- Large batch sizes are critical for performance to amortize the cost of parameter transfers and updates. Conditional computation reduces the batch sizes for the conditionally active chunks of the model.
 > [!note]
> We select a sparse number of experts for each input feature/token. 
> Therefore, the input tokens in a batch don't all go through the experts, and one expert may get $\frac{b}{n}$ input tokens in a batch of size b. More on this later.
>  
> Having a smaller batch size is bad when you consider the distributed setting. Not only does the LLM training perform worse with smaller batch sizes, but you also have lower efficiency in training time since you want to fill up the GPUs. If activation weights was 12GB/24GB, you want to end up with 12GB of inputs/backprop weights so that you efficiently use all your hardware.
- Additional loss terms are required to favor splitting up the computation between experts
	- This means beyond the regular [[Cross Entropy Loss]].
- At the time, not enough data (not large enough datasets) to train models with "millions, let alone billions of parameters".










##### Gating architecture: 
###### (Base, non-sparse) [[Softmax]] Gating
![[softmax_gating.png|100px]]
- Here we multiply the input by a trainable weight matrix $W_g$ and then apply softmax (i.e. single-layer linear layer or softmax classifier) 
- This insures that G(x) is a valid probability distribution with non-negative and sum = 1 properties.
###### (Sparse) Noisy Top-k Gating
The authors add two things to the gating: sparsity and noise.
- **H(x)** Before taking the softmax function, the authors add tuneable Gaussian noise, then keep only the topk values, setting the rest to -$\inf$
	- The point of the noise term is to help with load balancing across the experts.
	- The amount of noise added per component is a trainable weight matrix $W_{noise}$
![[Noisy Topk Gating.png|500]]
- Why softplus:
	- Noise must be positive
	- [[Softplus]] is smooth approx of ReLU, so it is differentiable

- Why use noise
	- Helps on initialization to balance the experts
	- Prevents over-utilization of experts
###### Model Initialization
- To initialize the model as approximately balanced (the soft constraints need some time to work), you can initialize the weight matrices $W_g$ and $W_{noise}$ as all 0s and thus the gating is dominated by the noise.
- This also helps to avoid out of memory errors with lots of dropped tokens.








##### Training the gating network:
- The gating network is trained through backprop like the rest of the model
- It is important to note that we only backpropogate through the chosen experts. Since argmax is not a differntiable function, this is an estimate of the overall gradient.
- Choosing k>1 (k experts) allows the gate values for the topk experts to have nonzero derivatives with respect to the weights of the gating network.

##### Shrinking Batch Problem
- Since you are splitting your batch over multiple experts, your batch size per expert is reduced to $\frac{kb}{n} << b$ which is less efficient and performant during learning especially as n increases.
- The solution is to make the original batch size as large as possible while still being able to store the activations between the forward and backward passes.
###### Solution: Mixing Data Parallelism with Model Parallelism:
- In conventional distributed training, multiple copies of the model are used asynchronously on different devices to process the distinct batches of data, the updated parameters are then synced through a set of parameter servers. 
- Here we: keep only one shared copy of each expert as a separate device in the distributed computing network. 
- Different batches are run synchronously for the rest of the model. 
- Therefore, if we distribute the base model over d devices, each with a batch size of b, each expert now gets $\frac{kbd}{n}$ examples.








##### Updated Loss function for balancing expert utilization
- Without additional loss terms, the gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing. **This is called routing collapse.**
- So we introduce a soft constraint on the batch-wise average on each gate, favoring a uniform distribution. The additional losses are:
	- $\mathbf{L}{importance}$: square of the coefficient of variation of the importance values
		- $\mathbf{G}(x)$ batch-wise sum of gate values for that expert
		- $\mathbf{w}_{importance}$: hand-tuned scaling factor
		- Goal: Encourages each expert to receive a roughly equal number of input tokens across the batch
	- $\mathbf{L}_{Load}$: Derivation:
		- This is important as it is what allows for the network to backpropogate through the hard topk decision for experts.
		- Here, they relax the hard topk decision boundaries to a probabilistic distribution
		- This is used for backpropogation to allow differentiate through a smooth prob dist, instead of discrete jumps through normal topk
		- $P(x, i) = Pr\left( \left( x \cdot W_g \right)_i + \text{StdNrm()} \cdot \text{Softplus}\left((x \cdot W_{noise})_i\right) > \text{kth\_excl}(H(x), k, i) \right)$
		- Explanation:
			- P(x,i) is the prob that the i-th expert is selection for input x. Depends on k-th highest gating score
			- $\left(x \times W_g\right)_i$ : Computes logits for all experts (i-th component is for expert i)
			- Softplus is the smooth approx of ReLU
		- $P(x, i) = \Phi\left(\frac{\left( x \cdot W_g \right)_i - \text{kth\_excluding}(H(x), k, i)}{\text{Softplus}\left((x \cdot W_{noise})_i\right)}\right)$
			- This simplifies the P(x, i) by standardizing the score computation
			- $\Phi$ is the CDF of the std norm distribution
		- $\text{Load}(X)_i = \sum_{x \in X} P(x, i)$
			- Measures the total load of expert i over the entire dataset in X
			- High load imbalance among experts is thus minimized on a per batch basis and over the entire dataset.






###### Determining the number of experts:
- The authors state that you need to set the number of experts to k>1 for the following reasons:
	- **Improved load balancing**: The gating mechanism becomes too deterministic which means the load balancing loss cannot effectively encourage a uniform distribution expert utilization.
	- **Better gradient flow:** the backpropogation will go through multiple experts. 
		- Gradients are less likely to vanish or explode when distributed across experts
		- Combining the outputs of multiple expects can create more robust representations.







##### MoE Evaluation run:
- Flat MoE are tested for 4, 32 and 256 experts
- Hierarchical MoEs are tested for 256, 1024, 4096 experts
![[PerplexityResults.png]]![[MoETests.png]]