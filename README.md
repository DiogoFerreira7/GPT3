Note: Check if there are any nice tips for read me / markdown to format it nicely

## LLM Journey

This project follows the principle of iteratively implementing small building blocks each coming closer to the original GPT3 (124M) implementation.

#### The following was the order of the implementations:
- Bigram model
- MLP
- Transformer
- Tokeniser
- GPT3

<hr>

### How to

#### Loading the pretrained model weights

Changes to make in the code - what values to put in according to the paper

#### Understanding the tokeniser

This [tokenisation website](https://tiktokenizer.vercel.app), examples of how the tokeniser works

#### Datasets

Here is an awesome repository you can use if you want to train your own model on a different dataset - make sure you choose Pre Training datasets for this model (PT) https://github.com/Zjh-819/LLMDataHub?tab=readme-ov-file

#### Training your own model

Calculations
Max steps



The original paper has quite conservative parameters especially warm up and learning rate that you can play around with

#### Evaluating

How to use the evaluation methods

<hr>

### Papers

The following papers were read and used to match the GPT3 implementation to its true origin, understand the separate components and optimise the model

Attention is all you need

Flash attention & flash attention 2

gpt 3 / 2

cuda paper explaining bfloat16

<hr>

### Changes & Optimisations

A lot of the following optimisations took advantage of kernel fusion, I found [this to be an interesting read](https://stackoverflow.com/questions/53305830/cuda-how-does-kernel-fusion-improve-performance-on-memory-bound-applications-on) and an easy way to understand it

**check that these were not already in the paper implementation**

**Changes**
- Using own implementation of tokeniser
- Using SlimPajama / FineWeb-Edu for training https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1 sample-10BT / sample-100BT
- Applying layer normalisation prior instead of post and not affecting residual stream
- Using a cosine learning rate scheduler with warm up
- Data randomly sampled without replacement to reduce overfitting during training

**Optimisations**
- Training can tolerate significantly lower precisions, we can go from 19.5 TFLOPS to 312 TFLOPS using an A100 by switching from FP32 TO FP16 and TF32 - TensorFloat-32 provides an 8x faster approximation - as long as you don't mind the loss in precision
- AdamW fused kernel - combining sequential steps into a singular kernel to optimise the memory access patterns
- Following the weight sharing scheme mentioned and preventing the double initialisation of the wte and ln tensors
- Using powers of 2 for most parameters
- Autocast (Automatic mixed precision) is a context manager / decorator that allows regions of the script to run in mixed precision - making sure to use bfloat16 instead of the reduced precision float16
- Preventing reinitialisation of wte and lm_head as they share the same tensor
- torch.compile
- Using FlashAttention which torch.compile

<hr>

### Future ideas & Improvements

Potential improvements includes:
[DDP (Distributed Data Parallel)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#:~:text=DistributedDataParallel%20(DDP)%20implements%20data%20parallelism,collective%20communications%20in%20the%20torch.).

Considerations for this improvement:
- Ensure that our gradient accumulation batch sizes are divisible by the ddp_world_size (the number of GPUs we have)
- Update our dataloader as we don't want any processes to get the same chunks of data, an easy way being we take in the rank of each process, the new positon would have to now be mini_batch_size * number_of_gpus so that they are strided out.
- Loss and gradients would have to be averaged across all processes, we can make our master / main process do the printing and logging information. 
- Logging would also have to take into account the number of GPUs processing especially in the tokens/second calculation.

Once pre trained it would be quite interesting to further take the model through fine-tuning stages that would allow it to interact in a conversational manner and being able to use RLHF. [OpenAI guidance is here](https://platform.openai.com/docs/guides/fine-tuning)

<hr>

### Example Outputs

#### Randomly initialised weights

- I am a doctor, let me teach you aboutVIS speaksposition Fund Pulitzer Recently astronautsumbnails tutorialFloat loneliness shift358 Woods calibr Doyleiven sedanzengroups licking Auschwitz mindful Tripoli 125

- I am a doctor, let me teach you about facilit debunk stating951 customizationLet indicatorlifting Jenn052 BAD ashamed antitTra scripting funny nihil Houth Marc Maiden vegetarian 33 Punjab manslaughter shipping

- I am a doctor, let me teach you aboutheet football Invention Congratulations Capitals transcriptsolding Railroadqua Steele HalSolution wee reboot Lebanon Panicersed testifiedARBinduced Getty Assets stretches relationships911

- I am a doctor, let me teach you about Conor acted=-=-=-=- exchanging scamsadier EngelsCar �dem carrying Puzzle productions439 brow trainthro insert Audio informingCentralruly chauscience 2000

#### Pretrained GPT2 Weights

- I am a doctor, let me teach you about the importance of mental health and family care," said the speech. "When you are an older man with mental illness, the

- I am a doctor, let me teach you about medicine. So, where do I live? I live here in the same building that you're in, and then I send

- I am a doctor, let me teach you about what I taught you and what I taught you before," Dr. Burdick continued. "I knew my specialty." For

- I am a doctor, let me teach you about the brain."When I said "no", he turned to face me without looking me right into his eyes.

#### Trained GPT Model

<hr>

### Common Problems & Fixes

Making sure cuda is installed, you can use torch.cuda.is_available(). If false is returned the following resources were quite useful for me in diagnosing those errors.
- [Comprehensive StackOverflow Guide](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)
- [Nvidia Cuda ToolKit Download](https://developer.nvidia.com/cuda-downloads)

<hr>

### Credits

OpenAI - [GPT-2 Tensorflow Implementation](https://github.com/openai/gpt-2/blob/master/src/model.py)

Hugging Face Tranformers - [GPT-2 PyTorch Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

pytorch

andrej Karpathy

