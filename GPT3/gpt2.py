from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import tiktoken
import numpy as np
from transformers import GPT2LMHeadModel
from math import sqrt, cos, pi

# TODO read through his git read me at the bottom for issues

# TODO make it different - add optional timing during inference for optimisation use time.time() and torch.cuda.synchronize() before the last one and check the difference

# TODO Change everything that I think could maek it better, e.g naming - how things work, check with gpt for different ways to write the same code

# TODO add dropout check where the paper added them

# TODO look into batch size scheduling in the gpt 3 paper and we can reinitialise the dataloader as an alternative - or just include this idea in readme

# TODO check the hugging face / open ai implementations for ideas of how to organise things

# TODO look for regularisation methods such as Dropout

# TODO go thorugh andrejs checkpoint saving and checkpoint resuming, this is important - makes it so model can be trained multiple days in a row and keep improving

# TODO instead of using tiktoken try using own tokeniser

# TODO update the dataset loading, we can use huggingface to download it first, then split both and update the dataloader
# streaming and using that loaded version to do training
# add randomness

# TODO After model training upload it to huggingface, see if inference can be done on it?

# TODO read the gpt 2 and the gpt 3 paper - read 4s improvements too and see if anything can be added and changed - gpt3 has more details for optimisations / training
    # Context length 2048, hyperparameters around transformer changed too in gpt3, 175 billion

# TODO Try EleutherAI for implementing custom models
# TODO https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage
    # Put results within the readme after
    # Check his code implementation he used for HellaSwag

# TODO Enable pylint and fix any errors / warnings that I can to make the code more readable

# TODO check his more optimised implementation in another repo and even the .c one for ideas

@dataclass
class GPTConfig:
    """
    Original GPT3 Paper Configuration
    block_size: int = 1024
    # 50,000 BPE merges, 256 byte tokens + <|endofline|>
    vocabulary_size: int = 50257
    number_of_layers: int = 12
    number_of_heads: int = 12
    number_of_embeddings: int = 768
    """
    block_size: int = 1024
    # Model becomes more efficient by using powers of 2
    vocabulary_size: int = 50304
    number_of_layers: int = 12
    number_of_heads: int = 12
    number_of_embeddings: int = 768

class GPT2(nn.Module):

    def __init__(self, config):
        # Initialising nn.Module base class for proper integration with parameters, gpu support, etc
        super().__init__()
        # Storing the config so we can easily define our hyper parameters in a @dataclass
        self.config = config

        """
        GPT2 architecture and naming conventions to follow

        transformer.wte.weight torch.Size([50257, 768])
        transformer.wpe.weight torch.Size([1024, 768])

        transformer.h.0.Block - Defined Below
        ...
        transformer.ln_f.weight torch.Size([768])
        transformer.ln_f.bias torch.Size([768])
        lm_head.weight torch.Size([50257, 768])
        """

        # Module Dict here allows us to index onto the sub modules using the keys / strings
        # Trying to follow the open ai naming convention for the layers
        self.transformer = nn.ModuleDict(dict(
            # Embeddings / Positional embeddings - just a wrapper around a tensor
            wte = nn.Embedding(config.vocabulary_size, config.number_of_embeddings),
            wpe = nn.Embedding(config.block_size, config.number_of_embeddings),
            # Hidden Layer - ModuleList just like module dict allows us to index but using integers instead of keys / strings
            h = nn.ModuleList([Block(config) for number_of_blocks in range(config.number_of_layers)]),
            # Final LayerNorm
            ln_f = nn.LayerNorm(config.number_of_embeddings),
        ))

        # Final classifier - language model head that will project 768 embedding dimension to the vocabulary size
        # The GPT paper does not use a bias for this contraction
        self.lm_head = nn.Linear(config.number_of_embeddings, config.vocabulary_size, bias=False)

        # We can tie our output weights and our input weights - we do not have to train as many parameters due to the weight sharing
        # We want our two matrices to behave similarly, if we have similarity between two tokens we wnat them to be nearby in the space
        # Both positons top and bottom by being tied improves performance - this was adopted in attention is all you need and in the GPT2 paper
        # This can also be seen in the openai gpt2 model.py
        # This way we set the pointers equal to eachother and we can share them - this is not the most efficient way as we can still not generate the wte embeddings initially
        self.transformer.wte.weight = self.lm_head.weight

        # TODO check how these layers are alraedy initialised
        self.apply(self._initialise_weights)

    def _initialise_weights(self, module):
        # TODO This is the default implementation if we are following the GPT2 source code - 1/root(number of features) - update it so that it is dynamic
        # TODO prevent the double initialisation of the linear and wte tensors
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # Every block of the residual network keeps adding and the variance keeps updating - we want to remove this growth
                # This can be done by updating the weights by 1/root(n) where n is the number of residual layers
                std *= (2 * self.config.number_of_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # If it is a linear layer that has bias then we can initialise them all to 0
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        # Tokens of shape (B, T)
        B, T = tokens.size()

        # Make sure that initially our sequence of tokens being forwarded is shorter than our block_size (context length)
        assert T <= self.config.block_size, f"Cannot forward a sequence of length {T}, the block size is only {self.config.block_size}"

        # Create initial position tensor of size T, and for efficiency reasons put it onto the cuda device
        # Indices of shape T - should be on the same device - by using tokens.device we prevent mismatches
        positions = torch.arange(0, T, dtype=torch.long, device = tokens.device)
        # Position embeddings will be identical for all rows and thus we can use broadcasting 
        positional_embeddings = self.transformer.wpe(positions)
        token_embeddings = self.transformer.wte(tokens)
        
        # Calculate the new embeddings based on the token and position embedding combination
        # These here are being broadcast as our positional embeddings are (T, number_of_embeddings)
        # But our token embeddings are (B, T, number_of_embeddings)
        x = positional_embeddings + token_embeddings

        # Now we have to forward it through all the hidden layer blocks
        for block in self.transformer.h:
            # Iteratively pass our embeddings through the forward of our block that will use causal self attention and mlp
            x = block(x)
        
        x = self.transformer.ln_f(x)
        # This final layer projects 768 embeddings into the vocabulary size - they are a softmax away from becoming probabilities
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # We have to flatten out the 3 dimensional tensor of logits - we need to pass both the logits and the targets
            # We flatten the logits from 3 dimensional (B, T, vocabulary_size) into (B*T, vocabulary_size) - the last dimension must match the targets dimension it can then broadcast it
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    # Make it so that the model does not have to be initialised to get a pretrained version
    @classmethod
    def from_pretrained(cls, model_type):
        print("Loading pretrained weights...")

        # TODO update this so that it just loads the GPT3 model
        configuration = {
            "number_of_layers": 12, 
            "number_of_heads": 12, 
            "number_of_embeddings": 768,
            # Below are constant for all models
            "vocabulary_size": 50257,
            "block_size": 1024
            }


        # Initialise our GPT model
        config = GPTConfig(**configuration)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Discard this buffer for the autoregressive mask
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # Initialise a HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        # Discard this buffer for the autoregressive mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        # Only for GPT2 since it was implemented in Tensorflow - these specific ones are annoyingly transposed so they have to
        # be reversed so that they fit with the PyTorch implementation
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # TODO if importing over the GPT3 ones then we might not need this we can just copy over the values without any checks
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # Return a model with the values copied over
        return model


class Block(nn.Module):
    # Typically layer normalisations are done post but it is desirable to do it without affecting our residual stream
    # Causal self attention has a reduction operation where we are doing weighted sums and MLP is a mapping function
    # Hence it is similar to a map, reduce operation
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        Block Definition from original paper

        transformer.h.0.ln_1.weight torch.Size([768])
        transformer.h.0.ln_1.bias torch.Size([768])

        transformer.h.0.Attention

        transformer.h.0.ln_2.weight torch.Size([768])
        transformer.h.0.ln_2.bias torch.Size([768])

        transformer.h.0.MLP
        """
        # Applying layer normalisation to stabilise and improve the performance of gradient propagation especially in deeper networks
        self.ln_1 = nn.LayerNorm(config.number_of_embeddings)
        self.attn = CausalSelfAttention(config)
        # Layer normalisation prior to MLP and residual layer
        self.ln_2 = nn.LayerNorm(config.number_of_embeddings)
        self.mlp = MLP(config)

    def forward(self, x):
        # Here we have a clean residual stream that does not have any layer normalisation applied meaning gradients from the top
        # flow directly through unchanged, they are desirable from an optimisation standpoint
        # hence why we apply layer normalisation directly to the attention and the mlp not the residual stream
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        MLP definition from paper

        transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
        transformer.h.0.mlp.c_fc.bias torch.Size([3072])

        GELU activation function

        transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
        transformer.h.0.mlp.c_proj.bias torch.Size([768])
        """

        self.c_fc = nn.Linear(config.number_of_embeddings, 4 * config.number_of_embeddings)
        # There are two versions the original and the approximate, RELU without a flat tail at 0
        # Gaussian Error Linear Units
        # Exact version is better the original change was due to an ERF function
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.number_of_embeddings, config.number_of_embeddings)
        # TODO see if there is another way to do NANOGPT SCALE INIT - maybe within the residual forward itself?
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module): 

    def __init__(self, config):
        super().__init__()
        assert config.number_of_embeddings % config.number_of_heads == 0

        """
        Causal self attention transformer definition from paper

        transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
        transformer.h.0.attn.c_attn.bias torch.Size([2304])
        transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
        transformer.h.0.attn.c_proj.bias torch.Size([768])
        """

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.number_of_embeddings, 3 * config.number_of_embeddings)
        # output projection
        self.c_proj = nn.Linear(config.number_of_embeddings, config.number_of_embeddings)
        # 
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.number_of_heads = config.number_of_heads
        self.number_of_embeddings = config.number_of_embeddings

    def forward(self, x):
        # Batch size, sequence length and number of embeddings
        B, T, C = x.size()

        # We have the embeddings for the query, key, values all stored together in a batch for efficiency reasons
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.number_of_embeddings, dim=2)

        # We are making the number of Heads into a batch dimension - it will apply operations on all in parallel
        # so that pytorch treats Heads as batches in parallel
        # Keys and querys
        # make sure that the number of heads * head size = to the number of channels/embeddings - nh*hs=C=768 channels in the Transformer
        k = k.view(B, T, self.number_of_heads, C // self.number_of_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.number_of_heads, C // self.number_of_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.number_of_heads, C // self.number_of_heads).transpose(1, 2) # (B, nh, T, hs)

        # Attention - creating the T x T matrix for all queries and keys
        # 3 methods - another one is by using lower triangular matrix and them summing across dim = 1

        # Flash Attention - the is_causal parameter removes our need to implement the masked_fill method
        # Flash attention can have up to a 7.6x faster speedup on computation based on the flash attention paper
        # It actually has more FLOPS and has to have an algorithmic rewrite hence it is not found by torch.compile
        # Flash attention 2 has come out
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Masked attention to prevent attending to future
        # # matrix multiply queries and keys and transpose the last two dimensions (the non batch dimensions)
        # y = (q @ k).transpose((-2, -1)) * (1.0 / sqrt(k.size(-1)))
        # # This is an autoregressive mask that prevents the values in the past from seeing into the future
        # y = y.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # # The softmax would normalise it so that all inf values go to 0 probability and get ignored
        # y = F.softmax(y, dim=1)
        # y = y @ v

        # Reassemble and perform a concatenation
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
# Creating our dataloader to get batches
class Dataloader:
    # TODO permute the shards / the documents - or add randomness to the data selected if the dataloader and loading method is changed
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        self.batch_size = B*T
        assert split in {"train", "val"}

        # Getting the shards
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [shard for shard in shards if split in shard]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, shard) for shard in shards]
        self.total_shards = len(self.shards)
        assert self.total_shards > 0, "no shards found"
        print(f"Retrieved {self.total_shards} shards")
        self.reset()

    def reset(self):
        self.current_shard = 0
        # Tokens here may have to be taken to the device
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.total_tokens = len(self.tokens)
        # Keep track of where in our data have previously loaded things
        self.position = 0

    def load_tokens(self, filename):
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self):
        B, T = self.B, self.T

        # We want our buffer to include the extra character so that we can easily get our targets using .view()
        # Once you cast any tensor to your device any following tensors that are using .view() will be within that device too
        dataset_buffer = self.tokens[self.position:self.position + (self.batch_size) + 1]
        x = dataset_buffer[:-1].view(B, T)
        targets = dataset_buffer[1:].view(B, T)

        # Update our position within the dataset, unless if that new position does not have enough data for another batch
        self.position += self.batch_size
        if self.position + (self.batch_size + 1) > self.total_tokens:
            self.current_shard = (self.current_shard + 1) % self.total_shards
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.position = 0
            self.total_tokens = len(self.tokens)

        return x, targets
    
class Trainer():

    def __init__(self):
        pass

# Device initialisation - the code will adapt to whatever device is being used
# using tokens.device within our forward to make sure that we place any other tensors that need computing within the same device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# torch.backend.mps.is_available() - apple silicone mps can be better than a cpu

# This is by default highest - in float32
# If we set this to "high" then it will use TensorFloat32 when it is available
# Medium with bfloat16 instead but it is no where near as precise
# A lot of the workloads in training as memory bound and thus even though we are supposed to get an 8x speed up it is bottlenecked and memory bound
# Note: Ampere / Turing architectures are required respectively
torch.set_float32_matmul_precision("high")

# model = GPT2.from_pretrained("gpt2")
# Initialising with the default config
model = GPT2(GPTConfig())

# During inference we have to make sure that we set the model to evaluation mode as BatchNorm and Dropout layers act different
# model.eval()
# Move every single tensors to cuda
model.to(device)

# There is no good reason to not use torch.compile in general, speed up is mainly from reducing python overhead and GPU read and writing
# GeLU non linearity - by compiling we know what instructions will be ran in order, e.g element operations for a variable will all the done at once whilst
# that memory is all on a GPU which prevents round trips - this is called kernel fusion
# model = torch.compile(model)

# 50,257 - we would hope that every element is getting roughly a uniform probability so that we are not confidently wrong - 1/50257
# Then if we want the negative loss likelihood (cross entropy) we can do -ln(1/50257)
# We want our loss to be roughly 10.8, in the model prior to pretraining it is 11.08

import time
# B = 16, T = 1024 is the default paper configuration
# Try to maximise the batch size that will fit your GPU without giving you out of memory errors (use even numbers)
# We want to get roughly a 0.5M batch size - so we can simulate a 0.5M batch size using gradient accummulation we just have to run for longer

# Nicest power of 2 closest to 0.5M (2^19)
total_batch_size = 2048 # 524288 
B = 1
# GPT3 has 2048 instead
T = 1024
mini_batch_size = B * T
assert total_batch_size % (mini_batch_size) == 0, "Total batch size is not divisible by the mini batches (B * T)"
# After this number of accumulation steps we can do one large update
gradient_accumulation_steps = total_batch_size // (mini_batch_size)
train_dataloader = Dataloader(B=B, T=T, split="train")
validation_dataloader = Dataloader(B=B, T=T, split="val")

# logits, loss = model(x, targets)

encoding = tiktoken.get_encoding("gpt2")

# TODO when all of these are organised go thorugh teh paper and make these calculations automatic e.g 10% min_lr (already is done) and others
# Aiming for 0.5M which is around 2^19 tokens per batch (we are using gradient accumulation of course), 
# we want 10^9 - 10B total tokens processed which is what we have. 10^9 / 2^19 = 19,073 batches roughly that we need to process all of it
max_learning_rate = 6e-4
min_learning_rate = max_learning_rate * 0.1
max_steps = 19073
# Gpt3 paper warms up over 375 million tokens, we have 0.5M per batch so we do this over 715 steps, 375e6 / 2^19 = 715 warm up steps
# 715 matches exactly but this is quite mild and we can probably start far earlier since we are limited on compute
warmup_steps = 715

# TODO learn adam and understand what the hyper parameters mean
# AdamW works well but we can try plenty of other ones
# Keeps momentum buffers (similar to RMSProp) this speeds up optimisation
# GPT3 used Adam with beta_1 = 0.9, beta_2 = 0.95 and e = 1e-8
# Fused by default is set to False to provide adequate bake in time as it is relatively new
# Instead of interating in a for loop and updating parameters which would launch lots of kernels, they aer all fused into a single kernel that updates them all
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True, weight_decay=0.1)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, min_learning_rate, max_learning_rate, warmup_steps)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, max_steps, min_learning_rate)
learning_rate_scheduler = torch.optim.lr_scheduler.SequentialLR(optimiser, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# TODO Tidy up this code - move each separate one into the class - maybe make a separate class that takes a model instead and lets you do sampling, evaluation, inference
for step in range(max_steps):

    # Evaluate our model a certain amount of steps
    if step % 100 == 0:
        print("Evaluating")
        model.eval()
        validation_dataloader.reset()
        with torch.no_grad():
            validation_loss_accumulation = 0.0
            validation_loss_steps = 20
            for _ in range(validation_loss_steps):
                x, targets = validation_dataloader.next_batch()
                x, targets = x.to(device), targets.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, targets)
                loss = loss / validation_loss_steps
                validation_loss_accumulation += loss.detach()
            
        print(f"Validation Loss: {validation_loss_accumulation.item():.5f}")

    # TODO THIS MAY ONLY SAMPLE WITHOUT TORCH.COMPILE, so if its on then don't sample if its off give user choice - TEST IF BROKEN OR NOT
    if step % 100 == 0:
        print("Sampling")
        # While the size of the rows that we will keep appending tokens to is smaller than our limit length
        model.eval()
        number_of_sequences = 4
        max_length = 35
        # Here we are just repeating the same start sentence tokens
        sentence_start = "I am a doctor, let me teach you about"
        # Get our prefix tokens given a string
        tokens = encoding.encode(sentence_start)
        # Flatten out the tensor and repeat it number of sequences times
        # TODO at the check check that Unsqueeze(0) shouldn't be needed here)
        tokens = torch.tensor(tokens).unsqueeze(0).repeat(number_of_sequences, 1)
        x = tokens.to(device)
        while x.size(1) < max_length:
            # We do not want to cache any tensors, no need to prepare for .backward() which will be more efficient
            with torch.no_grad():
                logits, loss = model(x)
                # Now we know each logit's position predicts the next position so if we want to generate a sentence we can 
                # take the last logit in each batch sequence - remember each logit is still of size vocabulary_size
                # TODO this implementation can be more efficient we are throwing away all other logits
                logits = logits[:, -1, :]
                # Now that we have the raw logits we take the softmax
                # We want to softmax the final dimension of size vocabulary_size to then sample from that distribution of probabilities
                probabilities = F.softmax(logits, dim = -1)
                # TODO check how sampling in the open ai tensorflow implementation works
                # This will clamp all other probabilities below 50 and thus it will keep the model within teh vicinity of likely tokens
                # They will all be renormalised and probabilities updated
                # topk_probabilities and the indices are all sorted in order by the topk function
                topk_probabilities, topk_indices = torch.topk(probabilities, 50, dim=-1)
                # After we get the top k highest probabilities we can sample from them
                # we then sample a value from them based on their probabilities giving us a (B, 1) tensor
                tokens = torch.multinomial(topk_probabilities, 1)
                # We are taking the second dimension and picking out the indices of the tokens that were sampled
                indices = torch.gather(topk_indices, -1, tokens)
                # Concatenate the new token index for each sentence with the current sentences x that we have
                # So we have a tensor of size [B, T] and [B, 1] to a [B, T+1]
                x = torch.cat((x, indices), dim=1)
                
        # All layers and modules are pre initialised randomly
        for i in range(number_of_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = encoding.decode(tokens)
            print(decoded)

    model.train()
    t0 = time.time()
    optimiser.zero_grad()
    loss_accumulation = 0.0
    for mini_batch_step in range(gradient_accumulation_steps):
        x, targets = train_dataloader.next_batch()
        x, targets = x.to(device), targets.to(device)

        # Automatic mixed precision - Autocast guidance
        # Autocast is a context manager / decorator that allows regions of the script to run in mixed precisoin
        # It will run in a dtype chosen by autocast to improve performance and maintain accuracy
        # Autocast should only wrap the forward pass of the network including hte loss computation, no backward passes are recommended using autocast
        # Warning: We do not want to use float16 otherwise we will need gradient scalars as they have a reduced range whilst bfloat16 has reduced precision
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # Mainly matrix multiplications are autocast to bfloat16
            # LayerNorm, softmax, logs will remain in float32 as they are susceptible to more precision changes
            logits, loss = model(x, targets)

        # TODO learn why gradient accumulation needs normalisation of this value
        # TODO this can be taken out as it will be a constant factor applied to everything
        # We need to make sure that we normalise the values and thus
        loss = loss / gradient_accumulation_steps
        # We need to detach the loss tensor, this is so that the tensor is not attached from the graph we just keep track of the values
        loss_accumulation += loss.detach()
        # Gradients will accumulate
        loss.backward()

    # TODO check if clipping slows it - we might be able to train quicker
    # Clipping gradients to have a maximum norm - calculates a global norm which makes sure that its length is no more than 1.0
    # Sometimes we can get unlucky batches and thus get very high loss which will prevent the model from having extremely large gradients
    global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

    # Update our learning rate according to the scheduler
    learning_rate_scheduler.step()

    # Wait for the GPU to complete otherwise it will time it before GPU is finished
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_per_second = (train_dataloader.B * train_dataloader.T * gradient_accumulation_steps) / (dt)
    # TODO add gradient accumulation step and epochs and the number of batches / accumulations per epoch - add the step value too out of total steps, calculate an estimate for how long it will take
    print(f"Current loss: {loss_accumulation.item()} - learning rate: {learning_rate_scheduler.get_last_lr()[0]:.4f}  - time: {dt*1000} - processing: {tokens_per_second} tokens/second")
