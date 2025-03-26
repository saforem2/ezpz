# Model Creation Guide

This guide provides a tutorial on how to create and use models with ezpz. We'll cover built-in models like MnistCNN and GPT, as well as instructions on how to integrate custom models into the ezpz framework.

## Table of Contents

1. [Built-in Models](#built-in-models)
   - [MnistCNN](#mnistcnn)
   - [GPT](#gpt)
2. [Creating Custom Models](#creating-custom-models)
3. [Integrating Models with ezpz](#integrating-models-with-ezpz)

## Built-in Models

ezpz provides several built-in models that you can use out of the box. Let's explore two of them: MnistCNN and GPT.

### MnistCNN

The MnistCNN is a Convolutional Neural Network designed for the MNIST dataset. Here's how you can use it:

```python
from ezpz.model import MnistCNN
import torch

# Create the model
model = MnistCNN()

# Example input (batch_size, channels, height, width)
x = torch.randn(1, 1, 28, 28)

# Forward pass
output = model(x)

print(output.shape)  # Should be (1, 10) for 10 classes
```

### GPT

ezpz also includes a GPT (Generative Pre-trained Transformer) implementation. Here's a basic example of how to use it:

```python
from wordplay.model import GPT
from wordplay.configs import ModelConfig

# Define the model configuration
model_config = ModelConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    bias=True,
    vocab_size=50257
)

# Create the model
model = GPT(model_config)

# Example input (batch_size, sequence_length)
x = torch.randint(0, model_config.vocab_size, (1, 100))

# Forward pass
logits, loss = model(x, x)

print(logits.shape)  # Should be (1, 100, vocab_size)
```

## Creating Custom Models

To create a custom model with ezpz, you need to subclass `torch.nn.Module`. Here's an example of a simple custom model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of your custom model
model = CustomModel(input_size=10, hidden_size=20, output_size=5)
```

## Integrating Models with ezpz

To integrate your model with ezpz, you'll typically need to create a Trainer class. Here's a simplified example based on the `TrainerMNIST` class from the provided code:

```python
import torch
from torch import nn, optim
from ezpz import get_rank, get_world_size, get_torch_device

class CustomTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.rank = get_rank()
        self.world_size = get_world_size(in_use=True)
        self.device = get_torch_device()
        
        self.model.to(self.device)
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
            )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def train_step(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                loss = self.train_step(data, target)
                if batch_idx % 100 == 0 and self.rank == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss}')

# Usage
model = CustomModel(input_size=784, hidden_size=128, output_size=10)
trainer = CustomTrainer(model, config)
trainer.train(train_loader, epochs=10)
```

This example demonstrates how to create a custom trainer that works with ezpz's distributed training capabilities. The trainer handles device placement, distributed data parallel wrapping, and the training loop.

Remember to adjust the model architecture, loss function, and optimizer according to your specific use case. The ezpz framework provides utilities for rank, world size, and device management, which are crucial for distributed training setups.

By following this guide, you should be able to create and use both built-in and custom models with the ezpz framework, leveraging its distributed training capabilities.