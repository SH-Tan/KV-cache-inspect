import torch
import torch.nn as nn

from utils.cuda_memory_utils import release_cuda_memory
# Import get_loaders function from data module within the same directory
from data_loaders.data import get_loaders 

import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    del testloader
    release_cuda_memory()
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    total_nll = 0.0
    loss_fct = nn.CrossEntropyLoss()
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        total_nll += float(neg_log_likelihood.detach().cpu())

        del inputs, lm_logits, shift_logits, shift_labels, loss, neg_log_likelihood
        if torch.cuda.is_available() and i % 50 == 0:
            torch.cuda.empty_cache()

    # Compute perplexity
    ppl = torch.exp(torch.tensor(total_nll / (nsamples * model.seqlen)))

    # Empty CUDA cache to save memory
    release_cuda_memory()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    total_nll = 0.0
    loss_fct = nn.CrossEntropyLoss()
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        total_nll += float(neg_log_likelihood.detach().cpu())

        del inputs, lm_logits, shift_logits, shift_labels, loss, neg_log_likelihood
        if torch.cuda.is_available() and i % 50 == 0:
            torch.cuda.empty_cache()

    # Compute perplexity
    ppl = torch.exp(torch.tensor(total_nll / (nsamples * model.seqlen)))

    # Empty CUDA cache to save memory
    release_cuda_memory()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["bangla","mmlu","hellaswag","winogrande","openbookqa","arc_easy"], 
        num_fewshot=0, use_accelerate=False, device = 'cuda'):
    from lm_eval import evaluator
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    
    # tm = TaskManager()
    # tasks = tm.all_tasks 
    
    # task_names = pattern_match(task_list, tasks)
    # def pattern_match(patterns, source_list):
    #     task_names = set()
    #     for pattern in patterns:
    #         for matching in fnmatch.filter(source_list, pattern):
    #             task_names.add(matching)
    #     return list(task_names)

    # task_names = pattern_match(task_list, TaskManager.all_tasks)
    
    task_names = list(task_list)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=device,
        limit=limit,
        check_integrity=False,
    )

    return results 
