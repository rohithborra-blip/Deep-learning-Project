"""
Configuration file for the deep learning project.
Centralized settings for easy management and reproducibility.
"""

class Config:
    """Base configuration class."""
    
    # Dataset
    DATASET = 'CIFAR10'  # 'CIFAR10' or 'MNIST'
    DATA_DIR = './data'
    
    # Model
    MODEL_TYPE = 'ImageClassifier'  # 'ImageClassifier' or 'SimpleCNN'
    NUM_CLASSES = 10
    DROPOUT_RATE = 0.5
    
    # Training
    BATCH_SIZE = 128
    VAL_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Optimizer and Scheduler
    OPTIMIZER = 'Adam'  # 'Adam', 'SGD'
    SCHEDULER = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau', 'CosineAnnealingLR'
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Device
    DEVICE = 'auto'  # 'auto', 'cuda', 'cpu'
    NUM_WORKERS = 2
    
    # Checkpointing
    CHECKPOINT_DIR = './checkpoints'
    SAVE_BEST_ONLY = True
    SAVE_FREQUENCY = 5  # Save every N epochs
    
    # Visualization
    VISUALIZATION_DIR = './visualizations'
    SAVE_VISUALIZATIONS = True
    
    # Paths
    RESULTS_FILE = 'results_summary.txt'
    
    # Data augmentation
    AUGMENT_TRAIN = True
    AUGMENT_VAL = False
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = True


class CIFARConfig(Config):
    """Configuration for CIFAR-10 dataset."""
    DATASET = 'CIFAR10'
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001


class MNISTConfig(Config):
    """Configuration for MNIST dataset."""
    DATASET = 'MNIST'
    NUM_CLASSES = 10
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'SimpleCNN'


class DevelopmentConfig(Config):
    """Configuration for development/testing."""
    DATASET = 'CIFAR10'
    BATCH_SIZE = 32
    EPOCHS = 2
    NUM_WORKERS = 0


def get_config(config_name='CIFAR10'):
    """Get configuration by name."""
    configs = {
        'CIFAR10': CIFARConfig,
        'MNIST': MNISTConfig,
        'DEV': DevelopmentConfig,
        'DEFAULT': Config
    }
    return configs.get(config_name, Config)


if __name__ == '__main__':
    # Display configuration
    cfg = get_config('CIFAR10')
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    for key, value in cfg.__dict__.items():
        if not key.startswith('_'):
            print(f"{key:.<40} {value}")
