import torch
from torch.nn import functional as F

import time
import tiktoken
from tqdm import tqdm

import pyo3_runtime

class Trainer():
    """
    Class for training, sampling and evaluation of a GPT3 model.

    Takes in a model and given an optimiser and schedule it will train the model for max_steps,
    and will evaluate and sample the model (by default given that sample=True and evaluate=True respectively),
    The sampling must take in a string which by default is "I am a doctor, let me teach you about",
    Optimisations have also been added including torch_compile and matmul_precision during training.
    Autocast will always be used during training wherever possible.

    Attributes:
        model (GPT3): GPT model initialised with the chosen GPTHyperParameters dataclass.
        optimiser (torch.nn.optim): By default the trainer will take in an AdamW optimiser, any should work.
        device ("cpu"/"cuda"): Choice of where the trainer will keep the tensors to train.
    """
    def __init__(self, model, optimiser, learning_rate_scheduler, device,
                 max_steps, gradient_accumulation_steps,
                 train_dataloader, evaluation_dataloader, tokeniser,
                 gradient_clipping=True, max_norm=1.0, train=True,
                 sample=True, sampling_string="I am a doctor, let me teach you about", sequences_to_sample=2, sampling_length=45, steps_per_sample=100,
                 evaluate=True, steps_per_evaluation=100,
                 torch_compile=True, matmul_precision="high",
                ):
        self.model = model
        self.optimiser = optimiser
        self.learning_rate_scheduler = learning_rate_scheduler
        self.device = device
        self.tokeniser = tokeniser

        # Loop
        self.max_steps = max_steps
        self.train = train
        
        # Sampling parameters
        self.sample = sample
        self.steps_per_sample = steps_per_sample
        self.sampling_string = sampling_string
        self.sequences_to_sample = sequences_to_sample
        self.sampling_length = sampling_length
        
        # Create the initial sequences that will be sampled
        self.sampling_tokens = self.tokeniser.encode(self.sampling_string)
        self.sampling_tokens = torch.tensor(self.sampling_tokens).repeat(self.sequences_to_sample, 1).to(self.device)

        # Evaluation parameters
        self.evaluate = evaluate
        self.steps_per_evaluation = steps_per_evaluation
        
        # Training parameters
        self.gradient_clipping = gradient_clipping
        self.max_norm = max_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Dataloaders
        self.train_dataloader = train_dataloader
        self.evaluation_dataloader = evaluation_dataloader

        # Logging variables
        self.loss = 0

        # Optimisations
        # There is no good reason to not use torch.compile in general, speed up is mainly from reducing python overhead and GPU read and writing
        # GeLU non linearity - by compiling we know what instructions will be ran in order, e.g element operations for a variable will all the done at once whilst
        # that memory is all on a GPU which prevents round trips - this is called kernel fusion
        if torch_compile:
            self.model = torch.compile(self.model)

        # This is by default highest - in float32, if we set this to "high" then it will use TensorFloat32 when it is available
        # Medium with bfloat16 instead but it is no where near as precise
        # A lot of the workloads in training as memory bound and thus even though we are supposed to get an 8x speed up it is bottlenecked and memory bound
        # Note: Ampere / Turing architectures are required respectively
        torch.set_float32_matmul_precision(matmul_precision)

    def start(self):
        for step in range(self.max_steps):

            # Training
            if self.train:
                start_time = time.time()
                self.training()
                # Wait for the GPU to complete before timing
                torch.cuda.synchronize()
                end_time = time.time()

                # Logging
                time_taken = end_time - start_time
                tokens_per_second = (self.train_dataloader.batch_size * self.train_dataloader.token_size * self.gradient_accumulation_steps) / (time_taken)
                predicted_time = int((self.max_steps - step) * time_taken)
                time_left_string = f"{predicted_time//3600}h:{(predicted_time%3600)//60}m:{predicted_time%60}s"
                print(f"Step {step}/{self.max_steps}, Loss: {self.loss} - LR: {self.learning_rate_scheduler.get_last_lr()[0]:.9f} - Time taken: {time_taken*1000} - Tokens/second: {tokens_per_second} - Time Remaining: {time_left_string}")

                # Evaluation
                if self.evaluate:
                    if step % self.steps_per_evaluation == 0:
                        print("\nEvaluating")
                        self.evaluation()
        
            # Sampling
            if self.sample:                    
                if step % self.steps_per_sample == 0:
                    print("\nSampling")
                    self.sampling()

    def training(self):
        self.model.train()
        self.optimiser.zero_grad()
        loss_accumulation = 0.0
        for _ in tqdm(range(self.gradient_accumulation_steps)):
            x, targets = self.train_dataloader.next_batch()
            x, targets = x.to(self.device), targets.to(self.device)

            # Automatic mixed precision - context manager / decorator that allows regions of the script to run in mixed precision
            # It will run in a dtype chosen by autocast to improve performance and maintain accuracy
            # Autocast should only wrap the forward pass of the network including hte loss computation, no backward passes are recommended using autocast
            # Warning: We do not want to use float16 otherwise we will need gradient scalars as they have a reduced range whilst bfloat16 has reduced precision
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                # Mainly matrix multiplications are autocast to bfloat16
                # LayerNorm, softmax, logs will remain in float32 as they are susceptible to more precision changes
                _, loss = self.model(x, targets)

            loss = loss / self.gradient_accumulation_steps
            # We need to detach the loss tensor, this is so that the tensor is not attached from the graph we just keep track of the values
            loss_accumulation += loss.detach()
            # Gradients will accumulate
            loss.backward()

        self.loss = loss_accumulation.item()

        # Clipping gradients to have a maximum norm - calculates a global norm which makes sure that its length is no more than 1.0
        # Sometimes we can get unlucky batches and thus get very high loss which will prevent the model from having extremely large gradients
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimiser.step()

        # Update our learning rate according to the scheduler
        self.learning_rate_scheduler.step()

    def sampling(self):
        # While the size of the rows that we will keep appending tokens to is smaller than our limit length
        self.model.eval()
        
        # This is the most method for cloning a tensor - we are doing this to prevent having to encode every time we want to sample
        # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
        sampling_tokens = self.sampling_tokens.detach().clone()
        while sampling_tokens.size(1) < self.sampling_length:
            # We do not want to cache any tensors, no need to prepare for .backward() which will be more efficient
            with torch.no_grad():
                logits, _ = self.model(sampling_tokens)
                # Each logit's position predicts the next position, take the last logit in each batch sequence - remember each logit is still of size vocabulary_size
                logits = logits[:, -1, :]
                # Now that we have the raw logits we take the softmax
                # We want to softmax the final dimension of size vocabulary_size to then sample from that distribution of probabilities
                probabilities = F.softmax(logits, dim = -1)
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
                sampling_tokens = torch.cat((sampling_tokens, indices), dim=1)

        for i in range(self.sequences_to_sample):
            try:
                tokens = sampling_tokens[i, :self.sampling_length].tolist()
                decoded = self.tokeniser.decode(tokens)
                print(decoded)
            except pyo3_runtime.PanicException as decoding_exception:
                print(decoding_exception)

    def evaluation(self):
        self.model.eval()
        self.evaluation_dataloader.reset()

        with torch.no_grad():
            validation_loss_accumulation = 0.0
            validation_loss_steps = 20
            for _ in range(validation_loss_steps):
                x, targets = self.evaluation_dataloader.next_batch()
                x, targets = x.to(self.device), targets.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(x, targets)
                loss = loss / validation_loss_steps
                validation_loss_accumulation += loss.detach()
            
        print(f"Validation Loss: {validation_loss_accumulation.item():.5f}")

if __name__ == "__main__":
    pass
