# Baisc imports
import os
import time
import logging

# Set env variables
os.environ['PYTHONHASHSEED'] = '1'
# You may need to set this variables by PyTorch. 
# It may limit overall performance so be aware of it.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

# Main imports
import random
import torch
import numpy as np
import hydra

# Additional imports
import torchvision
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf


# A logger for this file
log = logging.getLogger(__name__)


class Net(torch.nn.Module):
    """Net class for mnist example"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(
            torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        )
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def seed_worker(worker_id: int):
    """Set seed for Dataloader
    
    Args:
        worker_id (int) - CPU worker id
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(network: torch.nn.Module, 
          optimizer: torch.optim, 
          train_loader: torch.utils.data.DataLoader, 
          device: torch.device, 
          epoch: int):
    """Train the model

    Args:
        network (torch.nn.Module): model network
        optimizer (torch.optim): optimizer
        train_loader (torch.utils.data.DataLoader): data train loader
        device (torch.device): info about on which device to train model
        epoch (int): current epoch
    """
    network.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data.to(device))
        loss = torch.nn.functional.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
    
    end_time = time.time() - start            
    log.info(
        "Train Epoch: {} {:.4f}s\tLoss: {:.6f}".format(
            epoch,
            end_time,
            loss.item(),
            )
        )


def test(network: torch.nn.Module, 
          optimizer: torch.optim, 
          test_loader: torch.utils.data.DataLoader, 
          device: torch.device,
          g_train: torch.Generator,
          g_test: torch.Generator,
          save_path: str,
          epoch: int):
    """Test the model

    Args:
        network (torch.nn.Module): model network
        optimizer (torch.optim): optimizer
        test_loader (torch.utils.data.DataLoader): data test loader
        device (torch.device): info about on which device to train model
        g_train (torch.Generator): train Dataloader generator 
        g_test (torch.Generator): test Dataloader generator
        save_path (str): path where to save model dict
        epoch (int): current epoch
    """
    network.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            output = network(data.to(device))
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction='sum'
            ).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().detach().cpu()
    end_time = time.time() - start
    test_loss /= len(test_loader.dataset)
    log.info(
        "Test Epoch: {} {:.4f}s\tLoss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            epoch,
            end_time,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            )
        )

    # Save the model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            'torch_cuda_rng': 0 if not torch.cuda.is_available() else torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_state": random.getstate(),
            "generator_dataloader_train": g_train.get_state(),
            "generator_dataloader_test": g_test.get_state()
        },
        os.path.join(save_path, f"pytorch_model_{epoch}.pth"),
    )


@hydra.main(version_base=None, config_path="/mlcube/workspace", config_name="train")
def main(cfg : DictConfig) -> None:
    """Main training function
    
    Args:
     cfg (DictConfig): configs for the training.
    """
    # Info about device
    try:
        log.info("Devices we are using ", torch.cuda.get_device_name(), torch.cuda.device_count())
    except:
        log.info("Devices we are using is CPU :/")
        
    # Freeze and wrap cfg in more interesting package
    cfg = config_dict.FrozenConfigDict(OmegaConf.to_object(cfg))
    # Logging params
    log.info(cfg.params)
    
    # Creating dirs
    os.makedirs(cfg.output.data_path, exist_ok=True)
    os.makedirs(cfg.output.model_path, exist_ok=True)
    
    # Set seed
    random.seed(cfg.params.seed)
    np.random.seed(cfg.params.seed)
    torch.manual_seed(cfg.params.seed)
    torch.cuda.manual_seed(cfg.params.seed)

    # Gnerator for Dataloaders
    g_train = torch.Generator()
    g_train.manual_seed(cfg.params.seed)

    g_test = torch.Generator()
    g_test.manual_seed(cfg.params.seed)
    
    # Set running device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Set Datasets and Dataloaders
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            cfg.output.data_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=cfg.params.batch_size_train,
        shuffle=True,
        num_workers=cfg.params.num_workers,
        worker_init_fn=seed_worker, 
        generator=g_train,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            cfg.output.data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=cfg.params.batch_size_test,
        shuffle=False,
        num_workers=cfg.params.num_workers,
        worker_init_fn=seed_worker, 
        generator=g_test,
    )
    
    # We create a network and an optimizer
    network = Net().to(device)
    if cfg.params.optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.params.learning_rate)
    elif cfg.params.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(), lr=cfg.params.learning_rate, momentum=cfg.params.momentum)
    else:
        log.error("Sorry we only support: `adam` or `sgd`")
    
    # Train loop
    for epoch in range(1, cfg.params.n_epochs + 1):
        train(network, optimizer, train_loader, device, epoch)
        test(network, optimizer, test_loader, device, 
             g_train, g_test, cfg.output.model_path, epoch)
    
    

if __name__ == "__main__":
    # Deterministc variables
    # cudnn settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set rest of operation to be deterministic
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    # Run main function
    main()