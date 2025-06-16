# CLIP Implementation for CIFAR-10 Training

A PyTorch implementation of CLIP (Contrastive Language-Image Pre-training) specifically adapted for training on the CIFAR-10 dataset. This implementation includes a complete CLIP model architecture with both vision and text encoders, along with a training pipeline optimized for CIFAR-10.

## Features

- **Complete CLIP Architecture**: Includes both ResNet-based vision encoder and Transformer-based text encoder
- **CIFAR-10 Optimization**: Model architecture adapted for 32x32 images
- **Custom Text Templates**: Multiple text templates for data augmentation
- **BPE Tokenizer**: Simplified Byte-Pair Encoding tokenizer for text processing
- **Training Pipeline**: Full training loop with validation and model checkpointing
- **Visualization**: Training curves and inference testing

## Model Architecture

### Vision Encoder (ModifiedResNet)
- 3-layer stem convolution (instead of single layer)
- Anti-aliasing strided convolutions
- Attention pooling layer (QKV attention instead of average pooling)
- Bottleneck residual blocks with expansion factor of 4

### Text Encoder (Transformer)
- Multi-head self-attention blocks
- Custom LayerNorm for fp16 compatibility
- QuickGELU activation function
- Causal attention mask for autoregressive generation

### Key Components
- **Bottleneck**: ResNet bottleneck block with anti-aliasing
- **AttentionPool2d**: Spatial attention pooling for vision features
- **ResidualAttentionBlock**: Transformer block with residual connections
- **SimpleTokenizer**: BPE tokenizer for text processing

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
tqdm
ftfy
regex
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Prateeksingh-moa/clip-zero-shot-inference.git
cd clip-cifar10

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm ftfy regex
```

## Usage

### Training

Run the complete training pipeline:

```python
python CLIP.py
```

The training script will:
1. Download CIFAR-10 dataset automatically
2. Create a CLIP model optimized for CIFAR-10
3. Train for 50 epochs with cosine annealing scheduler
4. Save the best model checkpoint
5. Generate training curves visualization

### Model Testing

Test the trained model on sample images:

```python
# Uncomment the test function call at the bottom of the script
test_model_inference()
```

## Model Configuration

The default model configuration for CIFAR-10:

```python
CLIP(
    embed_dim=512,
    image_resolution=32,          # CIFAR-10 image size
    vision_layers=(2, 2, 2, 2),   # ResNet layer configuration
    vision_width=32,              # Base channel width
    vision_patch_size=None,       # Using ResNet, not ViT
    context_length=77,            # Text sequence length
    vocab_size=49408,             # BPE vocabulary size
    transformer_width=512,        # Text encoder width
    transformer_heads=8,          # Number of attention heads
    transformer_layers=6          # Number of transformer layers
)
```

## Training Details

### Data Augmentation
- Random horizontal flip
- Random crop with padding
- Normalization to [-1, 1] range

### Text Templates
Multiple templates are used for each class to increase text diversity:
- "a photo of a {}"
- "a picture of a {}"
- "an image of a {}"
- "a {} photo"
- "a {} picture"

### Optimization
- **Optimizer**: AdamW with learning rate 1e-4
- **Weight Decay**: 0.01
- **Scheduler**: Cosine annealing over 50 epochs
- **Batch Size**: 32
- **Loss Function**: Symmetric contrastive loss

### CIFAR-10 Classes
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

## File Structure

```
├── CLIP.py          # Core CLIP model implementation
└── README.md              # This file
```

## Key Functions

### Model Creation
- `create_model()`: Creates CLIP model optimized for CIFAR-10
- `build_model(state_dict)`: Builds model from state dictionary

### Training
- `train_epoch()`: Single epoch training loop
- `validate()`: Model validation with accuracy computation
- `contrastive_loss()`: Symmetric contrastive loss function

### Data Processing
- `CIFARCLIPDataset`: Custom dataset class for CIFAR-10 with text descriptions
- `collate_fn()`: Custom collate function for DataLoader
- `tokenize()`: Text tokenization using BPE

### Utilities
- `convert_weights()`: Convert model weights to fp16
- `test_model_inference()`: Test trained model on sample images

## Performance

The model achieves competitive performance on CIFAR-10 image-text matching tasks. Training typically converges within 30-40 epochs with the provided configuration.

### Expected Results
- **Training Loss**: Decreases from ~2.3 to ~0.5
- **Model Size**: ~50M parameters

## Customization

### For Different Datasets
1. Modify `image_resolution` in model config
2. Update `class_names` list
3. Adjust `vision_layers` and `vision_width` for complexity
4. Update text templates as needed

### Model Architecture
- Increase `embed_dim` for larger embedding space
- Adjust `transformer_layers` for text encoder depth
- Modify `vision_layers` for different ResNet configurations
- Change `transformer_width` for text encoder capacity

## Notes

- The BPE tokenizer creates a minimal vocabulary if the full BPE file is not available
- Model supports both fp16 and fp32 training
- Attention masks ensure proper causal modeling for text
- The implementation is optimized for educational purposes and may need optimization for production use

## References

- [Learning Transferable Visual Representations with CLIP](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP Repository](https://github.com/openai/CLIP)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License

This implementation is provided for educational and research purposes. Please refer to the original CLIP paper and OpenAI's implementation for licensing details.
