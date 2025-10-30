##### Authors of this paper:
- Two google engineers, and one author (not main contributor) that wrote the original [[Sparsely Gated MoE Layer]] layer paper in the [[Mixture of Experts (MoE)]].








##### Motivation:
The fun goal of this paper: train a model 1 trillion parameters (no one believed in this)
True motivation:
![[Switch Transformer Motivation.png]]

- As you can see, as you increase the number of experts the final test loss decreases. 
- Additionally the negative log [[Perplexity]] increase (better)
- So you want to be able to train with more parameters, but you don't want the computational cost to really increase.











##### Integration of MoE with Transformers:
- Instead of being applied convolutionally between stacked [[LSTMs]], you instead use the MoE to replace the FFNs with multiple FFNs (each one is an expert)
- This is easily integrated into the regular Vaswani [[Attention]] architecture
![[ModelDiagram_SwitchTransformer.png]]
- Note that the routing probabilities are used a scaling factor for the output of the FFN. So if it is confident in the expert choice, it contributes more to the output.
- They also use a soft-gating function (softmax probabilities) and then pick the highest gating probability. When backpropogating, they use the soft function to ensure differentiability

##### Simplification: Single expert routing
- The authors no longer use 2 experts, they now show that they can get accurate training with only 1 expert without losing on stability during training as was seen by all other previous papers.
- Benefits:
	- Improved hardware efficiency with simplified gating and routing between devices.
	- You can exchange money for accuracy. More experts does not increase computation since each expert is used independently, so you can run the experts fully in parallel on different devices and add as many experts as you want.







##### Expert routing with capacity:
![[ExpertCapacity.png]]

- If the expert is already at capacity (in red) that input token is dropped. You can otherwise increase the expert capacity, though that increases computation cost and is only scalable up to the max RAM on your GPU.







##### How they stabilize the model with k=1
![[Model_Types_SwitchTransfomrer.png]]

- Interesting note is that the authors say their training was stable for all models except the Switch-XXL, even though that one had the best accuracy on a per-step basis (so they don't train it fully).
- Other interesting side note is that as they increase the parameters, they reduce the number of transformer layers (24-> 15) but increase the number of experts (64->2048)









###### Dropout in gating mechanism:
- Instead of adding random noise to the output of the SGMoE, the authors introduce dropout to the gating system to avoid overfitting and improve regularization
- They do this mainly at the expert level, and can thus afford to have much higher dropout rates:
	- Overall dropout = 0.1
	- Expert dropout = 0.4
- This makes sense since the expert usage is already sparse so you can afford to dropout a large amount of parameters without losing too much of the model capability.
![[Switch_Dropout.png|700]]







###### Smaller parameter initialization for stability:
- The authors initialize the model parameters from a truncated Normal distribution with:
	- mean = 0
	- stdev = $\sqrt{s/n}$
		- s = scale hyper-parameter (they reduce it by a factor of 10 as compared to default transformer)
		- n = number of input units in the weight tensor (fan-in)









###### Selective precision:
- Model instability hinders training fully at BF16, so previous papers needed to train fully at the Pytorch default float32
- Data communications costs are too expensive with the whole model at float32
- So they selective cast the input tokens at each expert to float32, run the FFN and then cast it back to BF16 before it leaves the device.
- This is done through truncation since float32 just has more mantissa (fractional component) bits compared to brain-float 16. The range of values is the same. The first 7 are the same, but float32 allows for many more degrees of precision.
![[Float32_to_Bfloat16.png]]

- This allows for the training stability of float32, without incurring expensive communication costs.










##### Load balancing loss:
- As in SGMoE, the authors encourage balanced expert loading through an auxiliary loss. However, both of these losses are combined into one simplified loss term here.
- This loss favors a uniform distribution for the output distribution of the gating network
![[AuxiliaryLoss.png]]
- $f_i$ is how many tokens are routed to expert i, normalized by the number of tokens in the batch. Note that the $\mathbb{1}$ here is an indicator function (1 if True, 0 if False)
- $p_i(x)$ are probabilities of assigning a token to an expert from the gating mechanism as computed through a softmax function.
- $P_i$ is the average routing probability assigned to expert *i* across all tokens in the batch. Represents how much attention each router assigns to expert *i* for each batch
- Equation 4 is minimized under a uniform distribution where ,$f_i$ and $P_i$ are equal to $\frac{1}{N}$ so it encourages balancing the experts.

















##### Evaluation results:
![[Switch_Speedup.png]]

The Switch transformer also demonstrates improves in accuracy over all language training for log perplexity metric.

![[MultiLanguage.png]]


##### Parallelism strategies:
![[ParallelismStrat.png]]
- The diagram shows how to split up your data/model between devices to achieve parallelization.
- First diagram shows data parallelism -> Model is copied onto each device, data is split in batches across each device
- Model parallelism -> data is copied across each device, model is split and is thus very expensive since you have to run it sequentially
- Model and data parallelism -> Split data into mini-batches, and send through through sharded models.
- Expert and data parallelism -> data is split across devices, but each device takes one expert and doesn't need to communicate
- The last one is used for the 1T model, where even your expert doesn't fit on one device so you need to shard it across devices.
