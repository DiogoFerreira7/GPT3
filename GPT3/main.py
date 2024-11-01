import torch

from dataclasses import dataclass

from dataloader import Dataloader
from gpt import GPT3
from trainer import Trainer
from tokeniser import Tokeniser
import tiktoken

# Hyperparameters
@dataclass
class TrainerHyperParameters:
    """
    Try to maximise the batch size that will fit your GPU without giving you out of memory errors (powers of 2) - closest to 0.5M (2^19) as used by the GPT Paper.
    To match the paper using the EduFineWeb we wanted to process 10B (10^9) total tokens. 10^9 / 2^19 = 19,073 batches roughly that we need to process all of it
    """
    # Data 
    total_batch_size: int = 1048576 #, 524288, 262144
    batch_size: int = 16
    token_size: int = 2048

    # Training
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    max_learning_rate: float = 6e-4

    max_steps: int = 1000
    warmup_steps: int = 150

    # Calculations and assertions
    min_learning_rate = max_learning_rate * 0.1

    mini_batch_size = batch_size * token_size
    gradient_accumulation_steps = total_batch_size // (mini_batch_size)

    # Assertions
    assert total_batch_size % (mini_batch_size) == 0, "Total batch size is not divisible by the mini batches (B * T)"
    assert warmup_steps < max_steps, "Warm up steps must be less than the max steps you intend to train for"

@dataclass
class GPTHyperParameters:
    """
    Original GPT3 Paper Configuration
    block_size: int = 1024
    # 50,000 BPE merges, 256 byte tokens + <|endofline|>
    vocabulary_size: int = 50257
    number_of_layers: int = 12
    number_of_heads: int = 12
    number_of_embeddings: int = 768
    """
    block_size: int = 2048
    # Model might become more efficient by using powers of 2
    vocabulary_size: int = 50257
    number_of_layers: int = 24
    number_of_heads: int = 24
    number_of_embeddings: int = 2040

if __name__ == "__main__":
    # Device initialisation - the code will adapt to whatever device is being used using tokens.device within our forward to make sure that we place any other tensors that need computing within the same device
    # torch.backend.mps.is_available() - used for apple silicone mps
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    save_path = "Trained GPTs/GPT (1).model"
    # Choose between the default or pretrained model
    configuration = GPTHyperParameters()
    hyperparameters = TrainerHyperParameters()

    # Pre trained
    # model = GPT3.from_pretrained("gpt2", configuration)
    # model = GPT3.from_pretrained(save_path, configuration)
    # Brand new
    model = GPT3(configuration)
    # print(model.get_number_of_parameters())
    model.to(device)

    # Dataloader & Tokeniser initialisation
    # tokeniser = Tokeniser()
    # tokeniser.load("Tokeniser Vocabulary/shakespeare_tokeniser.tokeniser")
    tokeniser = tiktoken.get_encoding("gpt2")

    train_dataloader = Dataloader(hyperparameters.batch_size, hyperparameters.token_size, "train", tokeniser)
    evaluation_dataloader = Dataloader(hyperparameters.batch_size, hyperparameters.token_size, "val", tokeniser)

    # Optimiser and Schedulers
    # Fused by default is set to False to provide adequate bake in time as it is relatively new - Instead of interating in a for loop and updating parameters which would launch lots of kernels, they aer all fused into a single kernel that updates them all
    # GPT3 used Adam with beta_1 = 0.9, beta_2 = 0.95 and e = 1e-8
    optimiser = torch.optim.AdamW(model.parameters(), lr=hyperparameters.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True, weight_decay=hyperparameters.weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=0.01, total_iters=hyperparameters.max_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, hyperparameters.max_steps - hyperparameters.warmup_steps, hyperparameters.min_learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.SequentialLR(optimiser, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[hyperparameters.warmup_steps])

    # Trainer initialisation
    trainer = Trainer(model, optimiser, learning_rate_scheduler, device, 
                    hyperparameters.max_steps, hyperparameters.gradient_accumulation_steps, 
                    train_dataloader, evaluation_dataloader, tokeniser=tokeniser,
                    torch_compile=False,
                    train=False, evaluate=True, sample=True,
                    save=False, save_path=save_path,
                    wandb_logging=True)
    trainer.start()