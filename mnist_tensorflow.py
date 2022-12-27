# Baisc imports
import os
import time
import pickle
import logging

# Set env variables
os.environ['PYTHONHASHSEED'] = '0'

# Main imports
import tensorflow as tf
import hydra

# Additional imports
from omegaconf import DictConfig, OmegaConf


@tf.function
def train_step(model, loss_fn, optimizer, train_loss, train_accuracy, data, target):
    """Train function
    
    Args:
        model: keras model
        loss_fn: loss function
        optimizer: optimizer
        train_loss: training loss register
        train_accuracy: training accuracy register
        data: input data
        target: target for the input data
    """
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(target, predictions)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target, predictions)
    return


@tf.function
def test_step(model, loss_fn, test_loss, test_accuracy, data, target):
    """Train function
    
    Args:
        model: keras model
        loss_fn: loss function
        train_loss: testing loss register
        train_accuracy: testing accuracy register
        data: input data
        target: target for the input data
    """
    predictions = model(data, training=False)
    t_loss = loss_fn(target, predictions)
    test_loss(t_loss)
    test_accuracy(target, predictions)
    return


@hydra.main(version_base=None, config_path="/mlcube/workspace", config_name="train")
def main(cfg : DictConfig) -> None:
    """Main training function
    
    Args:
     cfg (DictConfig): configs for the training.
    """
    # Logging params
    logging.info(OmegaConf.to_yaml(cfg.params))
    
    # Set seed
    tf.keras.utils.set_random_seed(cfg.params.seed)    
    
    # Create a Tensorflow dataset / Keras dataset
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data(
        path=os.path.join(cfg.output.data_path, "mnist.npz")
    )
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Make train and test data loaders
    train_loader = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    test_loader = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(cfg.params.batch_size_test)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Create Keras model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Reshape((28,28,-1)),
            tf.keras.layers.Conv2D(filters=10, kernel_size=5),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=20, kernel_size=5),
            # sorry I did have small error when I used dropout here so I just removed it
            # It will not change enything just for comparison between architectures it will
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    # Compile model
    model.compile()
    
    # Define loss and optimizer
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if cfg.params.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.params.learning_rate)
    elif cfg.params.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=cfg.params.learning_rate, momentum=cfg.params.momentum)
    else:
        logging.error("Sorry we only support: `adam` or `sgd`")

    # Create checkpoint callback
    cp = tf.train.Checkpoint(model=model, optimizer=optimizer)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    
    # Train loop
    for epoch in range(1, cfg.params.n_epochs + 1):
        # Reset trackers for metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        dl_seed = cfg.params.seed + epoch

        # Train
        for idx, (images, labels) in enumerate(
                train_loader
                .shuffle(len(x_train), seed=dl_seed)
                .batch(cfg.params.batch_size_train)
        ):
            train_step(
                model, loss_fn, optimizer, train_loss, train_accuracy, 
                images, labels)
        
        # Save
        cp.write(os.path.join(cfg.output.model_path, f"tf_models_{epoch}.h5"))
        
        # Test
        for images, labels in test_loader:
            test_step(model, loss_fn, test_loss, test_accuracy, images, labels)

        # Log
        print(
            f"Epoch {epoch}, "
            f"Loss: {train_loss.result():.4f}, "
            f"Accuracy: {train_accuracy.result():.4f}, "
            f"Test Loss: {test_loss.result():.4f}, "
            f"Test Accuracy: {test_accuracy.result():.4f}"
        )


if __name__ == "__main__":
    # Deterministc variables
    # set the number of threads running on the CPU
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # set rest of operation to be deterministic
    tf.config.experimental.enable_op_determinism()

    # Info about device
    logging.info("Devices we are using")
    try:
        logging.info(tf.config.list_physical_devices("GPU"))
    except:
        logging.info("is CPU :/")
    
    # Run main function
    main()
    
    