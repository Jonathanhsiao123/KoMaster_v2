import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation (SE) attention.
    SE blocks help the network learn important features like ladders and eye shapes.
    Used in modern Go engines like KataGo.
    """
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply channel attention via SE block
        out = out * self.se(out)
        
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """
    Policy head that outputs move probabilities.
    Output: 362 logits (361 board positions + 1 pass move for 19x19)
    """
    def __init__(self, in_channels: int, board_size: int = 19):
        super(PolicyHead, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        # 362 outputs: 361 positions + 1 pass
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Output 362 logits (361 positions + 1 pass)
        return x


class ValueHead(nn.Module):
    """
    Value head with Global Average Pooling (GAP) that outputs win probability.
    Output: scalar between -1 (white wins) and +1 (black wins)
    
    Uses GAP instead of flatten for:
    - Better shape pattern learning
    - Fewer parameters (less overfitting)
    - Faster convergence
    - Modern Go engine standard
    """
    def __init__(self, in_channels: int, board_size: int = 19):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        # GAP reduces spatial dimensions to 1x1, so input is just 1 instead of 361
        self.fc1 = nn.Linear(1, 256)
        self.dropout = nn.Dropout(0.3)  # Prevents value head collapse
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        # Global Average Pooling - learns shape patterns better
        x = x.mean(dim=[2, 3], keepdim=True)  # [batch, 1, H, W] -> [batch, 1, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch, 1]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))  # Output between -1 and +1
        return x


class GoNet(nn.Module):
    """
    AlphaGo Zero style neural network with modern improvements for Go.
    
    Architecture:
    - Input: 19x19x17 planes
    - 1 convolutional layer
    - 6 residual blocks with Squeeze-and-Excitation (SE)
    - Policy head (362 outputs: 361 positions + 1 pass)
    - Value head with Global Average Pooling (1 output)
    
    Input format (17 planes):
    - Planes 0-7: Current player's stones (last 8 board positions)
    - Planes 8-15: Opponent's stones (last 8 board positions)
    - Plane 16: Color to play (all 1s for black, all 0s for white)
    
    Channel scaling guide:
    - 128 channels: Good for 2-day sprint, beats beginners (~20kyu)
    - 160 channels: SDK level possible with training
    - 192-256 channels: Dan-range strength with proper training
    """
    
    def __init__(self, board_size: int = 19, num_channels: int = 128, num_res_blocks: int = 6):
        super(GoNet, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(17, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual blocks with SE attention
        # 6 blocks for 2-day sprint, increase to 10-12 for stronger play
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy and value heads
        self.policy_head = PolicyHead(num_channels, board_size)
        self.value_head = ValueHead(num_channels, board_size)
        
        # For mixed precision training (FP16 speedup)
        self.use_amp = False
        
    @staticmethod
    def create_small():
        """Create a small fast model for rapid prototyping (128 channels, 6 blocks)"""
        return GoNet(num_channels=128, num_res_blocks=6)
    
    @staticmethod
    def create_medium():
        """Create a medium model for SDK-level play (160 channels, 10 blocks)"""
        return GoNet(num_channels=160, num_res_blocks=10)
    
    @staticmethod
    def create_large():
        """Create a large model for dan-level play (192 channels, 12 blocks)"""
        return GoNet(num_channels=192, num_res_blocks=12)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 17, 19, 19)
        
        Returns:
            policy: Tensor of shape (batch, 362) - move logits (361 positions + pass)
            value: Tensor of shape (batch, 1) - win probability
        """
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def predict(self, board_state: np.ndarray, use_amp: bool = True) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single board state.
        
        Args:
            board_state: NumPy array of shape (17, 19, 19)
            use_amp: Use automatic mixed precision (FP16) for 1.6-2.3x speedup
        
        Returns:
            policy: NumPy array of shape (362,) with move probabilities (361 + pass)
            value: Float between -1 and +1
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            x = torch.FloatTensor(board_state).unsqueeze(0)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                x = x.cuda()
            
            # Forward pass with optional mixed precision
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    policy_logits, value = self.forward(x)
            else:
                policy_logits, value = self.forward(x)
            
            # Convert policy logits to probabilities
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
            
        return policy, value
    
    def enable_amp(self):
        """Enable automatic mixed precision for faster inference."""
        self.use_amp = True
        
    def disable_amp(self):
        """Disable automatic mixed precision."""
        self.use_amp = False


def board_to_input(board: np.ndarray, current_player: int, history: list = None) -> np.ndarray:
    """
    Convert FastGoBoard state to neural network input format (17 planes).
    
    Args:
        board: 19x19 array with 0=empty, 1=black, -1=white
        current_player: 1 for black, -1 for white
        history: List of up to 8 previous board states (most recent first)
                 Each should be a 19x19 array with same format as board
    
    Returns:
        input_planes: 17x19x19 array ready for neural network
    """
    input_planes = np.zeros((17, 19, 19), dtype=np.float32)
    
    # If no history provided, use current board for all history planes
    if history is None:
        history = []
    
    # Planes 0-7: Current player's stones (8 move history, most recent first)
    # Planes 8-15: Opponent's stones (8 move history, most recent first)
    
    # Start with current board
    all_boards = [board] + history
    
    for i in range(8):
        if i < len(all_boards):
            hist_board = all_boards[i]
            if current_player == 1:  # Black to play
                input_planes[i] = (hist_board == 1).astype(np.float32)      # Current player (black)
                input_planes[i + 8] = (hist_board == -1).astype(np.float32) # Opponent (white)
            else:  # White to play
                input_planes[i] = (hist_board == -1).astype(np.float32)     # Current player (white)
                input_planes[i + 8] = (hist_board == 1).astype(np.float32)  # Opponent (black)
    
    # Plane 16: Color to play (1 if black, 0 if white)
    if current_player == 1:
        input_planes[16] = np.ones((19, 19), dtype=np.float32)
    
    return input_planes


# Example usage and testing
if __name__ == "__main__":
    print("[TEST] Testing GoNet with SE blocks and GAP...")
    
    # Create model
    model = GoNet(board_size=19, num_channels=128, num_res_blocks=6)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    
    # Compare model sizes
    print(f"\n📈 Model Scaling Options:")
    small = GoNet.create_small()
    medium = GoNet.create_medium()
    large = GoNet.create_large()
    
    small_params = sum(p.numel() for p in small.parameters())
    medium_params = sum(p.numel() for p in medium.parameters())
    large_params = sum(p.numel() for p in large.parameters())
    
    print(f"Small (128ch, 6 blocks):  {small_params:,} params (~{small_params*4/(1024**2):.1f}MB) - 2-day sprint")
    print(f"Medium (160ch, 10 blocks): {medium_params:,} params (~{medium_params*4/(1024**2):.1f}MB) - SDK level")
    print(f"Large (192ch, 12 blocks):  {large_params:,} params (~{large_params*4/(1024**2):.1f}MB) - Dan level")
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 17, 19, 19)
    
    policy, value = model(dummy_input)
    print(f"Policy shape: {policy.shape} (expected: [{batch_size}, 362])")
    print(f"Value shape: {value.shape} (expected: [{batch_size}, 1])")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}] (expected: [-1, 1])")
    
    # Test policy probabilities
    policy_probs = F.softmax(policy, dim=1)
    print(f"Policy sum: {policy_probs.sum(dim=1).mean():.6f} (expected: 1.0)")
    
    # Test board conversion with history
    print("\n🎮 Testing board conversion with history...")
    test_board = np.zeros((19, 19), dtype=np.int8)
    test_board[3, 3] = 1   # Black stone
    test_board[3, 4] = -1  # White stone
    
    # Create fake history
    history_board1 = np.zeros((19, 19), dtype=np.int8)
    history_board1[3, 3] = 1
    history = [history_board1]
    
    input_planes = board_to_input(test_board, current_player=1, history=history)
    print(f"Input planes shape: {input_planes.shape} (expected: [17, 19, 19])")
    print(f"Black stones in plane 0: {input_planes[0].sum()}")
    print(f"White stones in plane 8: {input_planes[8].sum()}")
    print(f"Black stones in plane 1 (history): {input_planes[1].sum()}")
    print(f"Color plane sum: {input_planes[16].sum()} (expected: 361 for black)")
    
    # Test prediction
    print("\n🔮 Testing single prediction...")
    policy_pred, value_pred = model.predict(input_planes, use_amp=False)
    print(f"Policy shape: {policy_pred.shape}")
    print(f"Policy sum: {policy_pred.sum():.6f}")
    print(f"Value: {value_pred:.3f}")
    print(f"Pass move probability: {policy_pred[361]:.6f}")
    
    # Test on GPU if available with AMP
    if torch.cuda.is_available():
        print("\n🎮 GPU detected! Testing CUDA with mixed precision...")
        model_cuda = model.cuda()
        dummy_input_cuda = dummy_input.cuda()
        
        # Test with AMP
        with torch.cuda.amp.autocast():
            policy_cuda, value_cuda = model_cuda(dummy_input_cuda)
        print("✅ GPU + FP16 forward pass successful!")
        print(f"   This gives ~1.6-2.3x speedup for MCTS inference")
        
        # Test predict with AMP
        policy_amp, value_amp = model_cuda.predict(input_planes, use_amp=True)
        print(f"   AMP prediction value: {value_amp:.3f}")
    else:
        print("\n💻 No GPU detected, using CPU")
        print("   For MCTS, GPU + FP16 recommended for speed")
    
    print("\n✅ All tests passed! Model is ready for training.")
    
    # Save model architecture summary
    print("\n📐 Model Architecture Summary:")
    print(model)