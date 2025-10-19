---
layout: blog
author: cohlem
---

This post is the continuation of this blog, where I experiment with the basics of distributed training. This post will explain how we apply those to RL training.

let's familiarize ourself with GRPO algorithm.

```python
model = initialize_model()
train_dataloader = initialize_train_dataloader(batch_size=per_rollout_size)

# Rollout step
for data in train_dataloader: # each len(data) will be per_rollout_size
	rollout_data = model.generate(data, responses_per_prompt=8) # we generate 8 responses per prompt for each entry inside data
	old_logprobs = calc_logprobs(rollout_data)
	entropy = calc_entropy(rollout_data)
	advantage = grpo_advantage(rollout_data)

	#update step
	update_dataloader = initialize_update_dataloader(batch_size=len(rollout_data)/update_per_rollout)
	for update_data in update_dataloader:
		logprobs = calc_logprobs(update_data)
		loss = grpo_loss(logprobs, old_logprobs)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

```

The generic algorithm would look something like above. In our case, we generate rollouts using SGLang, so the model we are using to generate rollouts and model we'll be using to do optimizer.step() will be different. So. we need to constantly update SGLang's model with the model that we just updated.

First let's initialize multiple processes and assign each process a GPU. The code below is straightforward and self-explanatory, cause it's a boilerplate code that we would use everywhere in distributed run.

```python

def setup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = 'nccl' if device == 'cuda' else 'gloo'
    rank = int(os.environ["RANK"])

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if device == 'cuda':
        torch.cuda.set_device(local_rank)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Initialize with explicit parameters
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank
    )
```

Now, we initialize a parent class for the actor (and also critic if ppo is used).

```python
class Worker:
    """
    This is the policy that we will be updating with each gradient update, we rollout using this policy's
    parameters, and we use the logprobs from this policy, we will also copy it's weights to make it old policy
    """

    def __init__(self, config):

#         self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.config = config
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        # first make a device mesh
        fsdp_size = int(int(os.environ['WORLD_SIZE']) / (config.ddp_size * config.tp_size))
        # this mesh will only be used for model partition
        self.mesh = init_device_mesh(device,(config.ddp_size,fsdp_size, config.tp_size), mesh_dim_names=["DDP", "FSDP", "TP"])
        self.dp_size = int(int(os.environ['WORLD_SIZE']) / self.config.tp_size)
        # this mesh will be used for data parallelism
        self.device_mesh = init_device_mesh(device,(self.dp_size, config.tp_size), mesh_dim_names=["DP", "TP"])

    def prepare_optimizer(self):
        self.model.gradient_checkpointing_enable()
        if self.config.tp_size > 1:
            self.model = prepare_tp_model(self.model, self.mesh)

        self.model = prepare_dp_model(self.model, self.mesh)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        # offload the model to cpu
        load_model_to_device(self, "cpu")

```

Notice, why we initialize two different device mesh one is `self.mesh` and another `self.device_mesh`. They have different purpose. `self.mesh` will be used for model
