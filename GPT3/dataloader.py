from datasets import load_dataset
import numpy as np
import torch

# Creating our dataloader to get batches
class Dataloader:
    def __init__(self, batch_size, token_size, split, tokeniser, num_streamed_examples=10):
        self.tokeniser = tokeniser
        self.batch_size = batch_size
        self.token_size = token_size
        self.total_batch_size = batch_size*token_size
        self.num_streamed_examples = num_streamed_examples
    
        assert split in {"train", "val"}
        self.my_iterable_dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

        # Reset the tracker
        self.reset()

    def reset(self):
        self.tokens = self.load_tokens()
        self.total_tokens = len(self.tokens)
        self.position = 0

    def load_tokens(self):
        tokens = [self.tokeniser._special_tokens['<|endoftext|>']]

        # Create a streamable iterator of the dataset
        data = iter(self.my_iterable_dataset.take(self.num_streamed_examples))
        for _ in range(self.num_streamed_examples):
            example = next(data)
            # Tokenise it and add it to the tokens array
            tokens.extend(self.tokeniser.encode(example["text"]))

        # Update the dataset and skip the ones already used
        self.my_iterable_dataset = self.my_iterable_dataset.skip(10)
        tokens_np = np.array(tokens)
        # Convert to a tensor and return
        return torch.tensor(tokens_np.astype(np.uint16), dtype=torch.long)

    def next_batch(self):
        # We want our buffer to include the extra character so that we can easily get our targets using .view()
        # Once you cast any tensor to your device any following tensors that are using .view() will be within that device too
        dataset_buffer = self.tokens[self.position:self.position + (self.total_batch_size) + 1]
        x = dataset_buffer[:-1].view(self.batch_size, self.token_size)
        targets = dataset_buffer[1:].view(self.batch_size, self.token_size)

        # Update our position within the dataset, unless if that new position does not have enough data for another batch
        self.position += self.total_batch_size
        if self.position + (self.total_batch_size + 1) > self.total_tokens:
            self.tokens = self.load_tokens()
            self.total_tokens = len(self.tokens)
            self.position = 0

        return x, targets
    
if __name__ == "__main__":
    pass
