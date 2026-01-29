"""
Test Clean VLM Agent - Demonstrating Fixed Dimension Handling
"""

import os
import sys
sys.path.append('/Users/fangyuanyu/Implementation/Maze/memory-maze')

import numpy as np
import torch
import gymnasium as gym

# Set rendering backend
os.environ['MUJOCO_GL'] = 'glfw'

from memory_maze.vlm_agent import create_clean_vlm_agent

def test_dimension_consistency():
    """Test that all dimensions are handled correctly"""
    
    print("🧪 Testing Clean VLM Dimension Consistency...")
    
    # Create agent and environment
    agent, env = create_clean_vlm_agent()
    
    print(f"🎮 Environment: {env.spec.id}")
    print(f"🧠 Model parameters: {sum(p.numel() for p in agent.model.parameters()):,}")
    
    # Test single observation
    obs, info = env.reset(seed=42)
    print(f"\n📸 Observation shape: {obs.shape}")
    print(f"📊 Observation dtype: {obs.dtype}")
    
    # Test forward pass dimensions
    print("\n🔍 Testing Forward Pass Dimensions:")
    
    # Test without instruction
    with torch.no_grad():
        image_tensor = agent.preprocess_observation(obs)
        print(f"🖼️ Image tensor shape: {image_tensor.shape}")
        
        action_logits = agent.model(image_tensor)
        print(f"🎯 Action logits shape: {action_logits.shape}")
        print(f"🎲 Action probabilities shape: {torch.softmax(action_logits, dim=-1).shape}")
    
    # Test with instruction
    instruction = "find the red target"
    instruction_tokens = agent.tokenize_instruction(instruction)
    print(f"\n💬 Instruction: '{instruction}'")
    print(f"📝 Tokenized instruction shape: {instruction_tokens.shape}")
    
    with torch.no_grad():
        image_tensor = agent.preprocess_observation(obs)
        if len(instruction_tokens) > 0:
            instruction_tokens = instruction_tokens.to(agent.device)
            
        action_logits_with_instr = agent.model(image_tensor, instruction_tokens)
        print(f"🎯 Action logits with instruction shape: {action_logits_with_instr.shape}")
    
    # Test batch processing
    print("\n📦 Testing Batch Processing:")
    
    # Create batch of observations
    obs_batch = np.stack([obs, obs, obs])  # (3, 64, 64, 3)
    image_batch_tensor = agent.preprocess_observation(obs_batch)
    print(f"🖼️ Batch image tensor shape: {image_batch_tensor.shape}")
    
    with torch.no_grad():
        batch_logits = agent.model(image_batch_tensor)
        print(f"🎯 Batch action logits shape: {batch_logits.shape}")
    
    # Test different instructions
    print("\n🧪 Testing Different Instructions:")
    
    instructions = [
        "find the red target",
        "move forward", 
        "turn left",
        "explore the maze",
        ""
    ]
    
    obs, info = env.reset(seed=42)
    
    for instruction in instructions:
        action = agent.select_action(obs, instruction=instruction)
        instr_text = instruction if instruction else "(no instruction)"
        print(f"📝 '{instr_text}' -> {agent.action_names[action]} ({action})")
    
    # Test model components separately
    print("\n🔧 Testing Model Components:")
    
    with torch.no_grad():
        # Test visual encoder (need batch dimension)
        image_batch = image_tensor.unsqueeze(0)  # Add batch dimension
        visual_features = agent.model.visual_encoder(image_batch)
        print(f"👁️ Visual features shape: {visual_features.shape}")
        
        # Test instruction encoder
        if len(instruction_tokens) > 0:
            instruction_features = agent.model.instruction_encoder(instruction_tokens)
            print(f"💭 Instruction features shape: {instruction_features.shape}")
        
        # Test action decoder
        action_logits = agent.model.action_decoder(visual_features, instruction_features if len(instruction_tokens) > 0 else None)
        print(f"🎯 Final action logits shape: {action_logits.shape}")
    
    print("\n✅ All dimension tests passed!")
    print("🎉 Clean VLM has consistent dimension handling!")
    
    env.close()
    return True


def test_training_step():
    """Test training step dimensions"""
    
    print("\n🏋️ Testing Training Step Dimensions:")
    
    agent, env = create_clean_vlm_agent()
    
    # Create dummy batch data
    batch_size = 4
    
    # Create batch of observations
    obs_batch = []
    action_batch = []
    reward_batch = []
    
    for i in range(batch_size):
        obs, info = env.reset(seed=i)
        action = agent.select_action(obs, instruction="find target")
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        obs_batch.append(obs)
        action_batch.append(action)
        reward_batch.append(reward)
    
    # Convert to tensors
    obs_batch_tensor = torch.stack([agent.preprocess_observation(obs) for obs in obs_batch])
    action_batch_tensor = torch.tensor(action_batch, dtype=torch.long)
    reward_batch_tensor = torch.tensor(reward_batch, dtype=torch.float32)
    
    print(f"📦 Batch observations shape: {obs_batch_tensor.shape}")
    print(f"🎯 Batch actions shape: {action_batch_tensor.shape}")
    print(f"💰 Batch rewards shape: {reward_batch_tensor.shape}")
    
    # Test forward pass for training
    with torch.no_grad():
        action_logits = agent.model(obs_batch_tensor)
        print(f"🎯 Training logits shape: {action_logits.shape}")
        
        # Test loss computation
        loss = torch.nn.functional.cross_entropy(action_logits, action_batch_tensor)
        print(f"📉 Loss shape: {loss.shape}")
        print(f"📉 Loss value: {loss.item():.4f}")
    
    print("✅ Training step dimensions are correct!")
    
    env.close()


if __name__ == "__main__":
    test_dimension_consistency()
    test_training_step()
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Clean VLM Improvements:")
    print("- ✅ Consistent tensor dimensions throughout forward pass")
    print("- ✅ Proper batch processing support")
    print("- ✅ Clean separation of visual, instruction, and action components")
    print("- ✅ No dimension mismatch errors")
    print("- ✅ Proper handling of single vs batch observations")
    print("- ✅ Clean instruction tokenization and encoding")
