import abc
import argparse
import json
import os
import random
from itertools import permutations
from typing import Set

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import IterableDataset

class SoftFold(nn.Module):
    """
    Sigmoid Fold module.

    This module performs a soft fold of the input data across the hyperplane orthogonal to the vector n 
    and going through the point n. It uses a sigmoid function to smoothly transition the folding effect.

    Attributes:
        n (nn.Parameter): The normal vector of the hyperplane (learnable parameter)
        crease (nn.Parameter or float): The sigmoid scaling factor (learnable or fixed)
        stretch (nn.Parameter or float): How far to fold the data over the hyperplane (learnable or fixed)
    """
    def __init__(self, width:int, crease:float=None, has_stretch:bool=True) -> None:
        """
        Parameters:
            width (int): The dimension of the input data
            crease (float, optional): A scaling factor for the sigmoid function. If None, it will be initialized as a learnable parameter.
            has_stretch (bool): Whether the module allows the stretch parameter to be learnable. Fixed at 2.0 if False.
        """
        super().__init__()
        # Hyperparameters
        self.width = width
        self.has_stretch = has_stretch
        
        # Initializing the n parameter
        n = torch.randn(self.width) * (2 / self.width) ** 0.5
        min_norm = 1e-2
        while n.norm().item() < min_norm:
            n = torch.randn(self.width) * (2 / self.width) ** 0.5
        self.n = nn.Parameter(n)

        # Initialize crease parameter
        if crease is None:
            std = 0.4
            self.crease = nn.Parameter(torch.randn(1) * std + 1) # sample from normal distribution centered at 1
        else:
            self.register_buffer('crease', torch.tensor(crease))
            
        # Initialize stretch as a parameter if needed
        if self.has_stretch:
            self.stretch = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_buffer('stretch', torch.tensor(2.0))
    
    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        """
        This function performs the soft folding operation on the input tensor.
        Parameters:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        Returns:
            output (torch.Tensor): The transformed tensor after the soft folding operation.
        """
        # pad the input if the width is greater than the input width, raise error if input width is greater than fold width
        if self.width > input_tensor.shape[1]:
            input_tensor = F.pad(input_tensor, (0, self.width - input_tensor.shape[1]))
        elif self.width < input_tensor.shape[1]:
            raise ValueError(f"Input dimension ({input_tensor.shape[1]}) is greater than fold width ({self.width})")

        # Compute x_dot_n, n_dot_n, and get scale_factor
        eps = 1e-8  
        x_dot_n = input_tensor @ self.n     # shape: (batch_size,)
        n_dot_n = self.n @ self.n + eps     # shape: (1,) add epsilon to avoid division by zero
        scale_factor = x_dot_n / n_dot_n    # shape: (batch_size,)

        # p caclulates how far away the input data is from the hyperplane, scaled by the crease parameter
        # positive p means the data is on the exterior of the hyperplane (if the side with the origin is 
        # considered the interior) and negative p means the data is on the interior of the hyperplane
        # the sigmoid variable helps to smooth the folding effect
        p = self.crease * (x_dot_n - n_dot_n)
        p = torch.clamp(p, min=-25.0, max=25.0)
        sigmoid = torch.sigmoid(p)          # shape: (batch_size,)

        # Get the difference between the input and its orthogonal projection onto the hyperplane
        # It's the offset that will be added to the input tensor stretch*sigmoid times to fold it 
        # over the hyperplane (2 times for an exact fold)
        residual_vec = (1 - scale_factor).unsqueeze(1) * self.n # shape: (batch_size, width)
        return input_tensor + self.stretch * sigmoid.unsqueeze(1) * residual_vec   # shape: (batch_size, width)

class AbstractDataset(abc.ABC):
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float):
        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ["o", "="] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        idxs = list(range(len(self.group_elements1) * len(self.group_elements2)))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = (
            idxs[: int(len(idxs) * frac_train)],
            idxs[int(len(idxs) * frac_train):],
        )

    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]

    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]

    def form_equation(self, a, b, c):
        return [a, "o", b, "=", c]

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)


class ModSumDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a + b) % self.p


class ModSubtractDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSubtractDataset, self).__init__(
            set(range(p)), set(range(p)), frac_train
        )
        self.p = p

    def fetch_output(self, a, b):
        return (a - b) % self.p


class ModDivisonDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModDivisonDataset, self).__init__(
            set(range(p)), set(range(1, p)), frac_train
        )
        self.p = p

    def fetch_output(self, a, b):
        return (a * pow(b, self.p - 2, self.p)) % self.p


class PermutationGroup(AbstractDataset):
    def __init__(self, k, frac_train):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])


class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {"train", "val"}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == "train":
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == "val":
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)


def operation_mod_p_data(operation: str, p: int, frac_train: float):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    if operation == "x_plus_y":
        data = ModSumDataset(p=p, frac_train=frac_train)
    elif operation == "x_minus_y":
        data = ModSubtractDataset(p=p, frac_train=frac_train)
    elif operation == "x_div_y":
        data = ModDivisonDataset(p=p, frac_train=frac_train)
    elif operation == "permutation":
        data = PermutationGroup(k=5, frac_train=frac_train)
    return data


def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    dataset = operation_mod_p_data(operation, prime, training_fraction)
    train_dataset = GroupDataset(dataset, "train")
    val_dataset = GroupDataset(dataset, "val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return (
        train_loader,
        val_loader,
        train_dataset.dataset.n_vocab,
        train_dataset.dataset.n_out,
    )


class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            SoftFold(dim_model * 4),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2


class Transformer(torch.nn.Module):
    def __init__(
            self,
            num_layers: int,
            dim_model: int,
            num_heads: int,
            vocab_size: int,
            output_size: int,
            seq_len: int,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, output_size),
        )

    def forward(self, inputs: Tensor):
        batch_size, context_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(
            torch.arange(context_len, device=inputs.device), "p -> b p", b=batch_size
        )
        position_embedding = self.position_embeddings(positions)

        embedding = token_embedding + position_embedding

        embedding = rearrange(embedding, "b s d -> s b d")

        return self.model(embedding)


def train(model, train_loader, optimizer, scheduler, device, num_train_batches):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    loss_total, correct = 0.0, 0.0
    total = 0

    # Loop over each batch from the training set
    count = 0
    for batch in train_loader:
        count += 1
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        correct += (torch.argmax(output, dim=1) == labels).sum()
        loss_total += loss * len(labels)
        total += len(labels)
        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()
        if count >= num_train_batches:
            break

    acc = correct / total
    loss = loss_total / total

    metrics = {
        "train_accuracy": float(acc),
        "train_loss": float(loss),
    }
    return metrics


def evaluate(model, val_loader, device, num_eval_batches):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.0
    total = 0
    count = 0
    # Loop over each batch from the validation set
    for batch in val_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1, :, :]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)
            total += labels.shape[0]
        count += 1
        if count >= num_eval_batches:
            break

    acc = correct / total
    loss = loss / total

    metrics = {"val_accuracy": float(acc), "val_loss": float(loss)}
    return metrics


def run(out_dir, dataset, seed_offset):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337 + seed_offset)
    train_loader, val_loader, n_vocab, n_output = get_data(
        operation=dataset,
        prime=97,
        training_fraction=0.5,
        batch_size=512,
    )

    model = Transformer(
        num_layers=2,
        dim_model=128,
        num_heads=4,
        vocab_size=n_vocab,
        output_size=n_output,
        seq_len=5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=0.5,
    )
    num_train_batches = 10
    num_eval_batches = 8
    num_total_updates = 7500
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(s / warmup_steps, 1)
    )

    final_info, train_log_info, val_log_info = [], [], []
    step_val_acc_99 = num_total_updates
    for ep in range(num_total_updates // num_train_batches):
        train_metrics = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            num_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_eval_batches,
        )
        train_metrics["step"] = (ep + 1) * num_train_batches
        val_metrics["step"] = (ep + 1) * num_train_batches

        if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
            step_val_acc_99 = val_metrics["step"]
        train_log_info.append(train_metrics)
        val_log_info.append(val_metrics)

    final_info = {
        "final_train_loss": train_metrics["train_loss"],
        "final_val_loss": val_metrics["val_loss"],
        "final_train_acc": train_metrics["train_accuracy"],
        "final_val_acc": val_metrics["val_accuracy"],
        "step_val_acc_99": step_val_acc_99,
    }
    print(final_info)
    with open(
            os.path.join(out_dir, f"final_info_{dataset}_{seed_offset}.json"), "w"
    ) as f:
        json.dump(final_info, f)
    return final_info, train_log_info, val_log_info


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

if __name__ == "__main__":
    num_seeds = {
        "x_div_y": 3,
        "x_plus_y": 3,
        "x_minus_y": 3,
        "permutation": 3,
    }

    out_dir = args.out_dir
    all_results = {}
    final_infos = {}
    for dataset in ["x_div_y", "x_minus_y", "x_plus_y", "permutation"]:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            print(f"Running {dataset} with seed offset {seed_offset}")
            final_info, train_info, val_info = run(args.out_dir, dataset, seed_offset)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)
        final_info_dict = {
            k: [d[k] for d in final_info_list] for k in final_info_list[0].keys()
        }
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items()}
        stderrs = {
            f"{k}_stderr": np.std(v) / len(v) for k, v in final_info_dict.items()
        }
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
