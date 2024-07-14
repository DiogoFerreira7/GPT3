import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import GPT2LMHeadModel

class GPT3(nn.Module):

    def __init__(self, config):
        # Initialising nn.Module base class for proper integration with parameters, gpu support, etc
        super().__init__()
        # Storing the config so we can easily define our hyper parameters in a @dataclass
        self.config = config

        """
        GPT3 architecture and naming conventions to follow

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
            h = nn.ModuleList([Block(config) for _ in range(config.number_of_layers)]),
            # Final LayerNorm
            ln_f = nn.LayerNorm(config.number_of_embeddings),
        ))

        # Final classifier - language model head that will project 768 embedding dimension to the vocabulary size
        # The GPT paper does not use a bias for this contraction
        self.lm_head = nn.Linear(config.number_of_embeddings, config.vocabulary_size, bias=False)

        # We can tie our output weights and our input weights - we do not have to train as many parameters due to the weight sharing
        # We want our two matrices to behave similarly, if we have similarity between two tokens we wnat them to be nearby in the space
        # Both positons top and bottom by being tied improves performance - this was adopted in attention is all you need and in the GPT3 paper
        # This can also be seen in the openai gpt2 model.py
        # This way we set the pointers equal to eachother and we can share them - this is not the most efficient way as we can still not generate the wte embeddings initially
        self.transformer.wte.weight = self.lm_head.weight

        # 50,257 - we would hope that every element is getting roughly a uniform probability so that we are not confidently wrong - 1/50257
        # Then if we want the negative loss likelihood (cross entropy) we can do -ln(1/50257)
        # We want our loss to be roughly 10.8, in the model prior to pretraining it is 11.08
        self.apply(self._initialise_weights)

    def _initialise_weights(self, module):
        # Apply a standard deviation of 0.02 as according to the paper
        std = 0.02
        # Every block of the residual network keeps adding and the variance keeps updating - we want to remove this growth
        if hasattr(module, "residual_initialisation_scaling"):
            # This can be done by updating the weights by 1/root(n) where n is the number of residual layers
            std *= (2 * self.config.number_of_layers) ** -0.5
    
        # Preventing double initialisation of the wte and lm_head since we are weight sharing
        if (isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)) and module is not self.lm_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # Linear layers have a bias too
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        # Tokens of shape (B, T)
        _, token_size = tokens.size()

        # Make sure that initially our sequence of tokens being forwarded is shorter than our block_size (context length)
        assert token_size <= self.config.block_size, f"Cannot forward a sequence of length {token_size}, the block size is only {self.config.block_size}"

        # Create initial position tensor of size T, and for efficiency reasons put it onto the cuda device
        # Indices of shape T - should be on the same device - by using tokens.device we prevent mismatches
        positions = torch.arange(0, token_size, dtype=torch.long, device = tokens.device)
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
        config = GPTHyperParameters(**configuration)
        model = GPT3(config)
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
        # Only for GPT3 since it was implemented in Tensorflow - these specific ones are annoyingly transposed so they have to
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
        self.residual_initialisation_scaling = True

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
        # Check if there is any residual initialisation
        self.residual_initialisation_scaling = True
        # regularization
        self.number_of_heads = config.number_of_heads
        self.number_of_embeddings = config.number_of_embeddings

    def forward(self, x):
        # Batch size, sequence length and number of embeddings
        batch_size, token_size, channel_size = x.size()

        # We have the embeddings for the query, key, values all stored together in a batch for efficiency reasons
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.number_of_embeddings, dim=2)

        # We are making the number of Heads into a batch dimension - it will apply operations on all in parallel
        # so that pytorch treats Heads as batches in parallel
        # Keys and querys
        # make sure that the number of heads * head size = to the number of channels/embeddings - nh*hs=C=768 channels in the Transformer
        k = k.view(batch_size, token_size, self.number_of_heads, channel_size // self.number_of_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(batch_size, token_size, self.number_of_heads, channel_size // self.number_of_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(batch_size, token_size, self.number_of_heads, channel_size // self.number_of_heads).transpose(1, 2) # (B, nh, T, hs)

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
        y = y.transpose(1, 2).contiguous().view(batch_size, token_size, channel_size)
        y = self.c_proj(y)
        return y