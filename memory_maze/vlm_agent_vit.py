"""
VLM Agent with ViT Backbone for Memory Maze Environment
Uses pre-trained ViT for visual encoding + transformer for action generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym

# Try to import torchvision models for ViT
try:
    from torchvision import models
    from torchvision.transforms import functional as F_transforms
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("Warning: torchvision not available, using mock ViT")


class ViTVisualEncoder(nn.Module):
    """ViT-based visual encoder using pre-trained backbone"""
    
    def __init__(self, feature_dim=256, patch_size=8, use_pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        
        if VIT_AVAILABLE and use_pretrained:
            # Load pre-trained ViT (small one)
            try:
                # Try to load ViT-B/16 or similar small model
                self.vit = models.vit_b_16(pretrained=True)
                self.vit.head = nn.Identity()  # Remove classification head
                
                # Get ViT feature dimension
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    vit_features = self.vit(dummy_input)
                    self.vit_dim = vit_features.shape[1]
                
                print(f"✅ Loaded pre-trained ViT-B/16 with {self.vit_dim} features")
                
                # Project ViT features to our desired dimension
                self.feature_proj = nn.Linear(self.vit_dim, feature_dim)
                
                # Resize input to 224x224 for ViT
                self.resize_transform = nn.AdaptiveAvgPool2d((224, 224))
                
            except Exception as e:
                print(f"Warning: Could not load pre-trained ViT: {e}")
                self._build_mock_vit()
        else:
            self._build_mock_vit()
    
    def _build_mock_vit(self):
        """Build a simple ViT-like encoder if pre-trained not available"""
        print("🔧 Building mock ViT encoder")
        
        # Simple patch embedding
        self.patch_embed = nn.Conv2d(3, 256, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Positional embeddings
        self.num_patches = (64 // self.patch_size) ** 2  # 8x8 = 64 patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, 256))
        
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Feature projection
        self.feature_proj = nn.Linear(256, self.feature_dim)
        self.vit_dim = 256
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 64, 64) or (batch, 3, 224, 224) for pre-trained ViT
        Returns:
            features: (batch, feature_dim)
        """
        if hasattr(self, 'vit'):  # Using pre-trained ViT
            # Resize to 224x224 for pre-trained ViT
            if x.shape[-2:] != (224, 224):
                x = self.resize_transform(x)
            
            # Get ViT features
            vit_features = self.vit(x)  # (batch, vit_dim)
            features = self.feature_proj(vit_features)  # (batch, feature_dim)
            
        else:  # Using mock ViT
            # Patch embedding
            patches = self.patch_embed(x)  # (batch, 256, 8, 8)
            patches = patches.flatten(2).transpose(1, 2)  # (batch, 64, 256)
            
            # Add positional embeddings
            patches = patches + self.pos_embed
            
            # Transformer encoding
            encoded = self.transformer(patches)  # (batch, 64, 256)
            
            # Global average pooling
            features = encoded.mean(dim=1)  # (batch, 256)
            features = self.feature_proj(features)  # (batch, feature_dim)
        
        return features


class SequenceBasedVLM(nn.Module):
    """Sequence-based VLM that treats vision as token sequence"""
    
    def __init__(self, feature_dim=256, num_actions=6, vocab_size=1000, max_seq_len=100, use_pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.max_seq_len = max_seq_len
        
        # Visual encoder (ViT)
        self.visual_encoder = ViTVisualEncoder(feature_dim=feature_dim, use_pretrained=use_pretrained)
        
        # Instruction tokenizer/embedding
        self.instruction_embedding = nn.Embedding(vocab_size, feature_dim)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.action_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # Main transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_actions)
        )
        
    def create_visual_sequence(self, visual_features):
        """Convert visual features to sequence format"""
        batch_size = visual_features.size(0)
        
        # Expand visual features to sequence
        visual_seq = visual_features.unsqueeze(1)  # (batch, 1, feature_dim)
        
        return visual_seq
    
    def create_instruction_sequence(self, instruction_tokens):
        """Convert instruction tokens to sequence"""
        if instruction_tokens is None:
            return None
            
        if instruction_tokens.dim() == 1:
            instruction_tokens = instruction_tokens.unsqueeze(0)
            
        batch_size = instruction_tokens.size(0)
        
        # Embed instruction tokens
        instruction_embeds = self.instruction_embedding(instruction_tokens)  # (batch, seq_len, feature_dim)
        
        # Truncate if too long
        if instruction_embeds.size(1) > self.max_seq_len - 2:  # Reserve space for CLS and ACTION tokens
            instruction_embeds = instruction_embeds[:, :self.max_seq_len-2]
        
        return instruction_embeds
    
    def forward(self, observation, instruction_tokens=None):
        """
        Forward pass with sequence-based processing
        
        Args:
            observation: RGB image (batch, 3, 64, 64) or single image (3, 64, 64)
            instruction_tokens: Optional instruction tokens (batch, seq_len) or (seq_len,)
        
        Returns:
            action_logits: (batch, num_actions) or (num_actions,)
        """
        # Handle single observation case
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False
        
        batch_size = observation.size(0)
        
        # Encode visual observation
        visual_features = self.visual_encoder(observation)  # (batch, feature_dim)
        visual_seq = self.create_visual_sequence(visual_features)  # (batch, 1, feature_dim)
        
        # Create instruction sequence
        instruction_seq = self.create_instruction_sequence(instruction_tokens)
        
        # Build target sequence with special tokens
        if instruction_seq is not None:
            # Combine: CLS + instruction + visual + ACTION
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            action_tokens = self.action_token.expand(batch_size, -1, -1)
            
            target_seq = torch.cat([
                cls_tokens,           # (batch, 1, feature_dim)
                instruction_seq,      # (batch, instr_len, feature_dim)
                visual_seq,           # (batch, 1, feature_dim)
                action_tokens         # (batch, 1, feature_dim)
            ], dim=1)                # (batch, total_len, feature_dim)
        else:
            # Simple: CLS + visual + ACTION
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            action_tokens = self.action_token.expand(batch_size, -1, -1)
            
            target_seq = torch.cat([
                cls_tokens,           # (batch, 1, feature_dim)
                visual_seq,           # (batch, 1, feature_dim)
                action_tokens         # (batch, 1, feature_dim)
            ], dim=1)                # (batch, 3, feature_dim)
        
        # Use visual features as memory (context)
        memory = visual_seq  # (batch, 1, feature_dim)
        
        # Decode sequence
        decoded_seq = self.transformer_decoder(
            tgt=target_seq,
            memory=memory
        )  # (batch, total_len, feature_dim)
        
        # Extract action token representation (last token)
        action_features = decoded_seq[:, -1, :]  # (batch, feature_dim)
        
        # Predict actions
        action_logits = self.action_head(action_features)  # (batch, num_actions)
        
        if single_obs:
            action_logits = action_logits.squeeze(0)  # (num_actions,)
        
        return action_logits


class ViTVLMAgent:
    """VLM Agent with ViT backbone"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", use_pretrained=True):
        self.device = device
        self.model = SequenceBasedVLM(use_pretrained=use_pretrained).to(device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded ViT-VLM model from {model_path}")
        else:
            print(f"Initialized new ViT-VLM model (pretrained={use_pretrained})")
            
        self.model.eval()
        
        # Action mapping for Memory Maze
        self.action_names = ["noop", "forward", "backward", "turn_left", "turn_right", "forward_left"]
        
        # Simple tokenizer for instructions
        self.vocab = {}
        self.vocab_size = 1000
        self._build_vocab()
        
    def _build_vocab(self):
        """Build simple vocabulary"""
        # Basic vocabulary
        words = ["find", "the", "red", "green", "blue", "target", "move", "forward", 
                "backward", "turn", "left", "right", "explore", "maze", "go", "to", 
                "look", "for", "search", "navigate", "walk", "run", "stop", "wait"]
        
        # Add character tokens
        for i in range(32, 127):  # Printable ASCII
            words.append(chr(i))
            
        # Build vocab dictionary
        for i, word in enumerate(words[:self.vocab_size]):
            self.vocab[word] = i + 1  # Reserve 0 for padding
            
    def tokenize_instruction(self, instruction: str) -> torch.Tensor:
        """Simple tokenization"""
        if not instruction:
            return torch.tensor([], dtype=torch.long)
            
        # Simple word-level tokenization
        tokens = []
        words = instruction.lower().split()
        
        for word in words:
            # Try to find exact word
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Try character-level tokenization
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                        
        return torch.tensor(tokens, dtype=torch.long)
        
    def preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Preprocess observation for ViT"""
        if isinstance(obs, dict):
            # Handle dict observations
            if 'image' in obs:
                image = obs['image']
            else:
                # Use first image-like observation
                for key, value in obs.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 3:
                        image = value
                        break
                else:
                    raise ValueError("No image found in observation dict")
        else:
            image = obs
            
        # Convert to tensor and normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Handle batch vs single image
        if len(image.shape) == 4:  # Batch of images: (batch, H, W, C)
            image = torch.from_numpy(image).permute(0, 3, 1, 2)  # BHWC -> BCHW
        elif len(image.shape) == 3:  # Single image: (H, W, C)
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
            
        return image.to(self.device)
    
    def select_action(self, obs: np.ndarray, instruction: Optional[str] = None) -> int:
        """Select action based on observation and optional instruction"""
        with torch.no_grad():
            # Preprocess observation
            image_tensor = self.preprocess_observation(obs)
            
            # Tokenize instruction if provided
            instruction_tokens = None
            if instruction:
                instruction_tokens = self.tokenize_instruction(instruction)
                if len(instruction_tokens) > 0:
                    instruction_tokens = instruction_tokens.to(self.device)
            
            # Get action logits
            action_logits = self.model(image_tensor, instruction_tokens)
            
            # Sample action
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
            return action


def create_vit_vlm_agent(env_name: str = "memory_maze:MemoryMaze-9x9-v0", 
                        model_path: Optional[str] = None, 
                        use_pretrained: bool = True) -> Tuple[ViTVLMAgent, gym.Env]:
    """Create ViT-VLM agent and environment"""
    
    # Create environment
    env = gym.make(env_name)
    
    # Create ViT-VLM agent
    agent = ViTVLMAgent(model_path=model_path, use_pretrained=use_pretrained)
    
    return agent, env
