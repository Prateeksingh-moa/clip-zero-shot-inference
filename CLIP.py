from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes,planes,1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes,planes,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes,planes*self.expansion,1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))


    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out
    
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self,d_model:int,n_head:int,attn_mask:torch.tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model,n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc",nn.Linear(d_model,d_model*4)),
            ("gelu",QuickGELU()),
            ("c_proj",nn.Linear(d_model*4,d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self,x:torch.tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype,device=x.device) if self.attn_mask is not None else None
        return self.attn(x,x,x,need_weights=False,attn_mask=self.attn_mask)[0]
    
    def forward(self,x:torch.tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self,width:int,layers:int,heads:int,attn_mask:torch.tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width,heads,attn_mask) for _ in range(layers)])

    def forward(self,x:torch.tensor):
        return self.resblocks(x)
    
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            # self.visual = VisionTransformer(
            #     input_resolution=image_resolution,
            #     patch_size=vision_patch_size,
            #     width=vision_width,
            #     layers=vision_layers,
            #     heads=vision_heads,
            #     output_dim=embed_dim
            # )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import List, Tuple
import random

# Import your CLIP model (assuming it's in a file called clip_model.py)
# from clip_model import CLIP, build_model

class CIFARCLIPDataset(Dataset):
    """
    Custom dataset that converts CIFAR labels to text descriptions
    """
    def __init__(self, cifar_dataset, class_names, transform=None):
        self.cifar_dataset = cifar_dataset
        self.class_names = class_names
        self.transform = transform
        
        # Create text templates for each class
        self.text_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a {} photo",
            "a {} picture",
        ]
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to text description
        class_name = self.class_names[label]
        
        # Randomly select a text template
        template = random.choice(self.text_templates)
        text_description = template.format(class_name)
        
        return image, text_description, label

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import List, Tuple
import random

# Import your CLIP model (assuming it's in a file called clip_model.py)
# from clip_model import CLIP, build_model

class CIFARCLIPDataset(Dataset):
    """
    Custom dataset that converts CIFAR labels to text descriptions
    """
    def __init__(self, cifar_dataset, class_names, transform=None):
        self.cifar_dataset = cifar_dataset
        self.class_names = class_names
        self.transform = transform
        
        # Create text templates for each class
        self.text_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a {} photo",
            "a {} picture",
        ]
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to text description
        class_name = self.class_names[label]
        
        # Randomly select a text template
        template = random.choice(self.text_templates)
        text_description = template.format(class_name)
        
        return image, text_description, label

# CLIP Tokenizer classes
import gzip
import html
from functools import lru_cache
import ftfy
import regex as re

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = None):
        # For demo purposes, we'll create a minimal BPE vocab if the file doesn't exist
        if bpe_path is None or not os.path.exists(bpe_path):
            self._create_minimal_vocab()
        else:
            self._load_vocab(bpe_path)
        
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def _create_minimal_vocab(self):
        """Create a minimal vocabulary for demo purposes"""
        # Only print this message once
        if not hasattr(SimpleTokenizer, '_vocab_created'):
            print("Creating minimal BPE vocabulary for demo...")
            SimpleTokenizer._vocab_created = True
            
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        
        # Add some common merges for English text
        common_merges = [
            ('t', 'h'), ('h', 'e'), ('i', 'n'), ('e', 'r'), ('a', 'n'),
            ('r', 'e'), ('e', 'd'), ('o', 'n'), ('e', 's'), ('n', 't'),
            ('t', 'i'), ('o', 'r'), ('o', 'u'), ('i', 't'), ('a', 't'),
            ('a', 's'), ('h', 'a'), ('n', 'g'), ('s', 't'), ('o', 'f')
        ]
        
        for merge in common_merges:
            vocab.append(''.join(merge))
            
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(common_merges, range(len(common_merges))))
        
    def _load_vocab(self, bpe_path):
        """Load vocabulary from BPE file"""
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

# Global tokenizer instance to avoid recreating it
_global_tokenizer = None

def get_tokenizer():
    """Get or create the global tokenizer instance"""
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = SimpleTokenizer()
    return _global_tokenizer

def tokenize(texts, context_length: int = 77):
    """
    Tokenize text using CLIP's tokenizer
    """
    if isinstance(texts, str):
        texts = [texts]
    
    tokenizer = get_tokenizer()  # Use global tokenizer instance
    
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    
    all_tokens = []
    for text in texts:
        tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
        
        # Pad or truncate to context_length
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token  # Ensure we end with eot_token
        else:
            tokens.extend([0] * (context_length - len(tokens)))
        
        all_tokens.append(tokens)
    
    return torch.tensor(all_tokens, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, texts, labels = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Tokenize texts using CLIP tokenizer
    tokenized_texts = tokenize(texts)
    
    # Convert labels to tensor
    labels = torch.tensor(labels)
    
    return images, tokenized_texts, labels

def contrastive_loss(logits_per_image, logits_per_text):
    """
    Compute contrastive loss for CLIP training
    """
    batch_size = logits_per_image.shape[0]
    
    # Create labels (diagonal should be the correct matches)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # Compute cross-entropy loss for both directions
    loss_img = nn.functional.cross_entropy(logits_per_image, labels)
    loss_text = nn.functional.cross_entropy(logits_per_text, labels)
    
    return (loss_img + loss_text) / 2

def create_model():
    """Create a CLIP model with reasonable parameters for CIFAR"""
    # Get tokenizer to determine vocab size
    tokenizer = get_tokenizer()  # Use global tokenizer
    vocab_size = len(tokenizer.encoder)
    
    model = CLIP(
        embed_dim=512,
        # Vision parameters
        image_resolution=32,  # CIFAR images are 32x32
        vision_layers=(2, 2, 2, 2),  # Smaller ResNet for CIFAR
        vision_width=32,  # Smaller width
        vision_patch_size=None,
        # Text parameters
        context_length=77,
        vocab_size=vocab_size,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=6
    )
    return model

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, texts, labels) in enumerate(pbar):
        images = images.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)
        
        # Compute loss
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, texts, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            texts = texts.to(device)
            
            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)
            
            # Compute loss
            loss = contrastive_loss(logits_per_image, logits_per_text)
            total_loss += loss.item()
            
            # Compute accuracy (image-to-text retrieval)
            predictions = logits_per_image.argmax(dim=1)
            correct_predictions += (predictions == torch.arange(len(predictions), device=device)).sum().item()
            total_samples += len(predictions)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize global tokenizer early
    print("Initializing tokenizer...")
    _ = get_tokenizer()  # This will create the tokenizer and print the message once
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    cifar_train = CIFAR10(root='./data', train=True, download=True, transform=None)
    cifar_val = CIFAR10(root='./data', train=False, download=True, transform=None)
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create custom datasets
    train_dataset = CIFARCLIPDataset(cifar_train, class_names, transform_train)
    val_dataset = CIFARCLIPDataset(cifar_val, class_names, transform_val)
    
    # Create data loaders
    batch_size = 32  # Small batch size for demo
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model
    print("Creating CLIP model...")
    model = create_model()
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, 'best_clip_model.pth')
            print("  -> Best model saved!")
        
        print("-" * 50)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")

def test_model_inference():
    """Test the trained model with a few examples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the best model
    model = create_model()
    checkpoint = torch.load('best_clip_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    cifar_test = CIFAR10(root='./data', train=False, download=False, transform=transform)
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Test with a few examples
    with torch.no_grad():
        for i in range(5):
            image, true_label = cifar_test[i]
            image = image.unsqueeze(0).to(device)
            
            # Create text queries for all classes
            text_queries = [f"a photo of a {name}" for name in class_names]
            tokenized_queries = tokenize(text_queries).to(device)
            
            # Get embeddings
            image_features = model.encode_image(image)
            text_features = model.encode_text(tokenized_queries)
            
            # Compute similarities
            similarities = (image_features @ text_features.T).squeeze()
            predicted_idx = similarities.argmax().item()
            
            print(f"Image {i+1}:")
            print(f"  True class: {class_names[true_label]}")
            print(f"  Predicted class: {class_names[predicted_idx]}")
            print(f"  Confidence: {similarities[predicted_idx]:.4f}")
            print()

if __name__ == "__main__":
    # main()
    
    # Test the model after training
    print("\nTesting trained model...")
    test_model_inference()