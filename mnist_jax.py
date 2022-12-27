# Baisc imports
import os
import time
import pickle
import logging

# Set env variables
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Main imports
import random
import torch
import jax
import numpy as np
import hydra

# Additional imports
import torchvision
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from omegaconf import DictConfig, OmegaConf


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x, training=False):
    x = nn.Conv(features=10, kernel_size=(5, 5))(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.relu(x)
    x = nn.Conv(features=20, kernel_size=(5, 5))(x)
    x = nn.Dropout(0.2)(x, deterministic=not training)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=50)(x)
    x = nn.relu(x)
    x = nn.Dropout(0.2)(x, deterministic=not training)
    x = nn.Dense(features=10)(x)
    return x


def cross_entropy_loss(logits, labels):
    """Compute the cross-entropy loss given logits and labels.

    Args:
        logits: output of the model
        labels: labels of the data
    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics_jax(logits, labels):
    """Compute metrics for the model.

    Args:
        logits: output of the model
        labels: labels of the data
    """
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


def create_train_state(rng, optimizer_name, learning_rate, momentum):
    """Creates initial `TrainState` with `sgd` or 'adam'."""
    cnn = CNN()
    rng, rng_dropout = jax.random.split(rng)
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
    if optimizer_name == "adam":
        tx = optax.adam(learning_rate)
    elif optimizer_name == "sgd":
        tx = optax.sgd(learning_rate, momentum)
    else:
        logging.error("Sorry we only support: `adam` or `sgd`")
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)


@jax.jit
def train_step(state, image, label, dropout_rng):
  """Train for a single step. Also jit-compiled for speed.
  
  Args:
    params: parameters of the model
    image: input image
    label: label of the image
    dropout_rng: rng for dropout
  """
  def loss_fn(params):
    logits = CNN().apply({'params': params}, x=image, training=True, rngs={'dropout': dropout_rng})
    loss = cross_entropy_loss(logits=logits, labels=label)
    return loss, logits
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics_jax(logits=logits, labels=label)
  return state, metrics


@jax.jit
def eval_step(params, image, label):
  """Evaluate for a single step. Also jit-compiled for speed.
  
  Args:
    params: parameters of the model
    image: input image
    label: label of the image
  """
  logits = CNN().apply({'params': params}, image)
  return compute_metrics_jax(logits=logits, labels=label)


def train_epoch(state, train_ds, epoch, rng):
  """Train for a single epoch.
  
  Args:
    params: model parameters
    test_ds: test dataloader
    epoch: current epoch
  """
  batch_metrics = []
  start = time.time()
  for batch_idx, (data, target) in enumerate(train_ds):
    rng, rng_dropout = jax.random.split(rng)
    state, metrics = train_step(state, data, target, rng_dropout)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]} # jnp.mean does not work on lists

  end_time = time.time() - start
  logging.info('train epoch: %d, time %.4f,, loss: %.4f, accuracy: %.2f' % (
      epoch, end_time, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
  return state, rng


def eval_model(params, test_ds):
  """Evaluate the model.
  
  Args:
    params: model parameters
    test_ds: test dataloader
  """
  batch_metrics = []
  start = time.time()
  for batch_idx, (data, target) in enumerate(test_ds):
    metrics = eval_step(params, data, target)
    batch_metrics.append(metrics)
    
  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  metrics = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]} # jnp.mean does not work on lists
  end_time = time.time() - start
  summary = jax.tree_util.tree_map(lambda x: x.item(), metrics) # map the function over all leaves in metrics
  return summary['loss'], summary['accuracy'], end_time


def image_to_numpy(img):
    """Transformations applied on each image => bring them into a numpy array
    Args:
        img: input image
    """
    img = np.array(img, dtype=np.float32)
    img = np.transpose(img, (1, 2, 0))
    return img


def numpy_collate(batch):
    """We need to stack the batch elements
    
    Args:
        batch: batch of data and target
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def seed_worker(worker_id: int):
    """Set seed for Dataloader
    
    Args:
        worker_id (int) - CPU worker id
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@hydra.main(version_base=None, config_path="/mlcube/workspace", config_name="train")
def main(cfg : DictConfig) -> None:
    """Main training function
    
    Args:
     cfg (DictConfig): configs for the training.
    """
    # Logging params
    logging.info(OmegaConf.to_yaml(cfg.params))
    
    # Set seed
    random.seed(cfg.params.seed)
    np.random.seed(cfg.params.seed)
    torch.manual_seed(cfg.params.seed)
    rng = jax.random.PRNGKey(cfg.params.seed)

    # Gnerator for Dataloaders
    g_train = torch.Generator()
    g_train.manual_seed(cfg.params.seed)

    g_test = torch.Generator()
    g_test.manual_seed(cfg.params.seed)

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
    
    # Initialize model and get params
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng,
                               cfg.params.optimizer, 
                               cfg.params.learning_rate, 
                               cfg.params.momentum)
    
    # Train loop
    for epoch in range(1, cfg.params.n_epochs + 1):
        state, rng = train_epoch(state, train_loader, epoch, rng)
        # Evaluate on the test set after each training epoch
        test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
        logging.info(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
            epoch, end_time, test_loss, test_accuracy * 100))
        # Save the model parameters
        numpy_rng = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.get_rng_state() # well only this is required but for the sake of completeness I am adding all
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.join("jax_checkpoints_{}".format(epoch)),
            target={"state": state,
                    "epoch": epoch,
                    "rng": rng,
                    "pytorch_rng": torch_state.numpy(),
                    "numpy_rng": numpy_rng,
                    "python_state": python_state,
                    "generator_dataloader_train": g_train.get_state().numpy(),
                    "generator_dataloader_test": g_test.get_state().numpy(),
                    },
                    step=epoch,
                    overwrite=True,
            )


if __name__ == "__main__":
    # Info about device
    logging.info("Devices we are using")
    try:
        logging.info(jax.devices())
    except:
        logging.info("is CPU :/")
    
    # Run main function
    main()