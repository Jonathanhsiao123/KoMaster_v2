import pygame
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from enum import Enum

# Initialize Pygame
pygame.init()

# Game States
class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3

# Constants
BOARD_SIZE = 19
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
BOARD_SIZE_PX = 640
GRID_SIZE = BOARD_SIZE_PX // (BOARD_SIZE - 1)
STONE_RADIUS = GRID_SIZE // 2 - 2
BOARD_OFFSET_X = (WINDOW_WIDTH - BOARD_SIZE_PX) // 2
BOARD_OFFSET_Y = 140

# Colors
BG_COLOR = (252, 248, 240)
BOARD_COLOR = (218, 165, 32)
LINE_COLOR = (0, 0, 0)
BLACK_STONE = (0, 0, 0)
WHITE_STONE = (255, 255, 255)
HOVER_COLOR = (218, 165, 32, 100)
TEXT_COLOR = (101, 67, 33)
BUTTON_COLOR = (255, 193, 37)
BUTTON_HOVER = (255, 179, 0)
AI_THINKING_COLOR = (255, 100, 100)
AI_CHOICE_COLOR = (100, 255, 100, 120)
MENU_BG = (245, 235, 220)
TITLE_COLOR = (139, 69, 19)

# Fonts
FONT_TITLE = pygame.font.Font(None, 72)
FONT_SUBTITLE = pygame.font.Font(None, 48)
FONT_LARGE = pygame.font.Font(None, 36)
FONT_MEDIUM = pygame.font.Font(None, 28)
FONT_SMALL = pygame.font.Font(None, 24)


class ResidualBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation (SE) attention"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        out += residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """Policy head that outputs move probabilities"""
    def __init__(self, in_channels, board_size=19):
        super(PolicyHead, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """Value head with Global Average Pooling"""
    def __init__(self, in_channels, board_size=19):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.mean(dim=[2, 3], keepdim=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        return x


class GoNet(nn.Module):
    """AlphaGo Zero style neural network"""
    def __init__(self, board_size=19, num_channels=128, num_res_blocks=6):
        super(GoNet, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        
        # Initial convolutional layer (17 input planes)
        self.conv_input = nn.Conv2d(17, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy and value heads
        self.policy_head = PolicyHead(num_channels, board_size)
        self.value_head = ValueHead(num_channels, board_size)
        
    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value



    
class AIPlayer:
    def __init__(self, model_path='checkpoints/model_pretrained1.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CHANGED: Try loading checkpoint first to detect architecture
        # WHY: Checkpoint contains large model (12 blocks, 192 filters)
        #      but code was hardcoded to small model (6 blocks, 128 filters)
        
        self.model = None
        self.top_moves = []
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Detect architecture from checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Count residual blocks by checking highest index
                max_block_idx = 0
                for key in state_dict.keys():
                    if 'res_blocks.' in key:
                        block_idx = int(key.split('res_blocks.')[1].split('.')[0])
                        max_block_idx = max(max_block_idx, block_idx)
                
                num_res_blocks = max_block_idx + 1
                
                # Detect number of channels from conv_input weight shape
                num_channels = state_dict['conv_input.weight'].shape[0]
                
                print(f"📊 Detected architecture: {num_res_blocks} blocks, {num_channels} filters")
                
                # Create model with detected architecture
                self.model = GoNet(
                    board_size=19,
                    num_channels=num_channels,
                    num_res_blocks=num_res_blocks
                ).to(self.device)
                
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ AI model loaded successfully from {model_path}")
                
            else:
                # Old checkpoint format (direct state dict)
                # Try to detect architecture the same way
                state_dict = checkpoint
                
                max_block_idx = 0
                for key in state_dict.keys():
                    if 'res_blocks.' in key:
                        block_idx = int(key.split('res_blocks.')[1].split('.')[0])
                        max_block_idx = max(max_block_idx, block_idx)
                
                num_res_blocks = max_block_idx + 1
                num_channels = state_dict['conv_input.weight'].shape[0]
                
                print(f"📊 Detected architecture: {num_res_blocks} blocks, {num_channels} filters")
                
                self.model = GoNet(
                    board_size=19,
                    num_channels=num_channels,
                    num_res_blocks=num_res_blocks
                ).to(self.device)
                
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ AI model loaded successfully from {model_path}")
                
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")
            print("⚠️  AI will use random moves")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  AI will use random moves")
            self.model = None
    
    def board_to_input(self, board, current_player):
        """Convert board state to 17-plane input tensor."""
        input_planes = np.zeros((17, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        # Convert board format
        board_array = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == 'black':
                    board_array[i][j] = 1
                elif board[i][j] == 'white':
                    board_array[i][j] = -1
        
        current_player_val = 1 if current_player == 'black' else -1
        
        # Plane 0: Current player's stones
        # Plane 8: Opponent's stones
        if current_player_val == 1:
            input_planes[0] = (board_array == 1).astype(np.float32)
            input_planes[8] = (board_array == -1).astype(np.float32)
        else:
            input_planes[0] = (board_array == -1).astype(np.float32)
            input_planes[8] = (board_array == 1).astype(np.float32)
        
        # History planes (simplified - use current position)
        for i in range(1, 8):
            input_planes[i] = input_planes[0]
            input_planes[i + 8] = input_planes[8]
        
        # Plane 16: Color to play
        if current_player_val == 1:
            input_planes[16] = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        return torch.FloatTensor(input_planes).unsqueeze(0).to(self.device)
    
    def get_move(self, game):
        """Get AI's next move with top choices stored for visualization"""
        self.top_moves = []  # Clear previous top moves
        
        if self.model is None:
            return self.get_random_move(game)
        
        with torch.no_grad():
            board_tensor = self.board_to_input(game.board, game.current_player)
            policy_logits, value = self.model(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            # Get valid moves with probabilities
            moves = []
            for idx in range(BOARD_SIZE * BOARD_SIZE):
                row = idx // BOARD_SIZE
                col = idx % BOARD_SIZE
                if game.board[row][col] is None:
                    moves.append(((row, col), policy_probs[idx]))
            
            # Sort by probability
            moves.sort(key=lambda x: x[1], reverse=True)
            
            # Store top 5 for visualization
            self.top_moves = moves[:5]
            
            # Debug print
            print(f"\n🤖 AI thinking... Top moves:")
            for i, ((r, c), prob) in enumerate(self.top_moves):
                print(f"   {i+1}. ({r}, {c}): {prob:.4f}")
            
            # Try top moves until legal
            for (row, col), prob in moves[:20]:
                temp_board = [row[:] for row in game.board]
                temp_captures = game.black_captures, game.white_captures
                original_player = game.current_player
                
                if game.place_stone(row, col):
                    # Legal move found, restore and return
                    game.board = temp_board
                    game.black_captures, game.white_captures = temp_captures
                    game.current_player = original_player
                    game.last_move = None
                    print(f"✅ AI chose: ({row}, {col}) prob={prob:.4f}")
                    return row, col
                else:
                    game.board = temp_board
                    game.black_captures, game.white_captures = temp_captures
                    game.current_player = original_player
            
            print("⚠️  No legal move in top 20, using random")
            return self.get_random_move(game)
    
    def get_random_move(self, game):
        """Fallback random move"""
        empty_positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if game.board[i][j] is None:
                    empty_positions.append((i, j))
        
        if empty_positions:
            import random
            move = random.choice(empty_positions)
            print(f"🎲 Random move: {move}")
            return move
        return None


# class GoBoard:
#     def __init__(self):
#         self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
#         self.current_player = 'black'
#         self.black_captures = 0
#         self.white_captures = 0
#         self.last_move = None
#         self.consecutive_passes = 0
#         self.move_history = []
        
#     def get_grid_pos(self, row, col):
#         """Convert board position to screen coordinates"""
#         x = BOARD_OFFSET_X + col * GRID_SIZE
#         y = BOARD_OFFSET_Y + row * GRID_SIZE
#         return x, y
    
#     def get_board_pos(self, mouse_x, mouse_y):
#         """Convert screen coordinates to board position"""
#         col = round((mouse_x - BOARD_OFFSET_X) / GRID_SIZE)
#         row = round((mouse_y - BOARD_OFFSET_Y) / GRID_SIZE)
        
#         if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
#             return row, col
#         return None, None
    
#     def get_group(self, row, col, color, visited=None):
#         """Get all stones in a group using flood fill"""
#         if visited is None:
#             visited = set()
        
#         key = (row, col)
#         if key in visited:
#             return []
#         if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
#             return []
#         if self.board[row][col] != color:
#             return []
        
#         visited.add(key)
#         group = [(row, col)]
        
#         for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             group.extend(self.get_group(row + dr, col + dc, color, visited))
        
#         return group
    
#     def has_liberties(self, row, col, color):
#         """Check if a group has any liberties"""
#         group = self.get_group(row, col, color)
        
#         for r, c in group:
#             for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nr, nc = r + dr, c + dc
#                 if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
#                     if self.board[nr][nc] is None:
#                         return True
#         return False
    
#     def remove_captures(self, row, col, opponent_color):
#         """Remove captured opponent stones"""
#         captured = 0
        
#         for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             nr, nc = row + dr, col + dc
#             if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
#                 if self.board[nr][nc] == opponent_color:
#                     if not self.has_liberties(nr, nc, opponent_color):
#                         group = self.get_group(nr, nc, opponent_color)
#                         for gr, gc in group:
#                             self.board[gr][gc] = None
#                             captured += 1
        
#         return captured
    
#     def place_stone(self, row, col):
#         """Attempt to place a stone"""
#         if self.board[row][col] is not None:
#             return False
        
#         # Place stone temporarily
#         self.board[row][col] = self.current_player
        
#         # Remove captures
#         opponent = 'white' if self.current_player == 'black' else 'black'
#         captured = self.remove_captures(row, col, opponent)
        
#         # Check suicide
#         if not self.has_liberties(row, col, self.current_player) and captured == 0:
#             self.board[row][col] = None
#             return False
        
#         # Update state
#         if self.current_player == 'black':
#             self.black_captures += captured
#         else:
#             self.white_captures += captured
        
#         self.last_move = (row, col)
#         self.move_history.append((row, col, self.current_player))
#         self.current_player = 'white' if self.current_player == 'black' else 'black'
#         self.consecutive_passes = 0
#         return True
    
#     def pass_turn(self):
#         """Pass the current turn"""
#         self.current_player = 'white' if self.current_player == 'black' else 'black'
#         self.last_move = None
#         self.consecutive_passes += 1
#         self.move_history.append(('pass', None, self.current_player))
    
#     def is_game_over(self):
#         """Check if game is over (two passes)"""
#         return self.consecutive_passes >= 2
    
#     def reset(self):
#         """Reset the game"""
#         self.__init__()
class GoBoard:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'black'
        self.black_captures = 0
        self.white_captures = 0
        self.last_move = None
        self.consecutive_passes = 0
        self.move_history = []
        self.ko_point = None  # Track forbidden ko point
        self.previous_board_state = None  # Track previous board for ko detection

    def copy_board_state(self):
        """Return an immutable snapshot of the board (tuple of tuples)."""
        # Use tuple of tuples so comparisons are cheap and reliable
        return tuple(tuple(cell for cell in row) for row in self.board)

    def boards_equal(self, a, b):
        """Compare two board snapshots (tuple-of-tuples)."""
        return a == b

    def get_grid_pos(self, row, col):
        """Convert board position to screen coordinates"""
        x = BOARD_OFFSET_X + col * GRID_SIZE
        y = BOARD_OFFSET_Y + row * GRID_SIZE
        return x, y

    def get_board_pos(self, mouse_x, mouse_y):
        """Convert screen coordinates to board position"""
        col = round((mouse_x - BOARD_OFFSET_X) / GRID_SIZE)
        row = round((mouse_y - BOARD_OFFSET_Y) / GRID_SIZE)

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None, None

    def get_group(self, row, col, color, visited=None):
        """Get all stones in a group using flood fill"""
        if visited is None:
            visited = set()

        key = (row, col)
        if key in visited:
            return []
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return []
        if self.board[row][col] != color:
            return []

        visited.add(key)
        group = [(row, col)]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            group.extend(self.get_group(row + dr, col + dc, color, visited))

        return group

    def get_group_liberties(self, group):
        """Return a set of liberties (empty points) adjacent to a group."""
        liberties = set()
        for (r, c) in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.board[nr][nc] is None:
                        liberties.add((nr, nc))
        return liberties

    def has_liberties(self, row, col, color):
        """Check if a group has any liberties"""
        group = self.get_group(row, col, color)

        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.board[nr][nc] is None:
                        return True
        return False

    def remove_captures(self, row, col, opponent_color):
        """Remove captured opponent stones adjacent to (row,col). Return captured positions."""
        captured_positions = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if self.board[nr][nc] == opponent_color:
                    if not self.has_liberties(nr, nc, opponent_color):
                        group = self.get_group(nr, nc, opponent_color)
                        for gr, gc in group:
                            # Remove captured stones
                            self.board[gr][gc] = None
                            captured_positions.append((gr, gc))

        return captured_positions

    def place_stone(self, row, col):
        """Attempt to place a stone with proper ko rule enforcement"""
        if self.board[row][col] is not None:
            return False

        # Check ko rule: can't play at forbidden ko point
        if self.ko_point == (row, col):
            print(f"⚠️ Ko rule violation prevented at {(row, col)}")
            return False

        # Save current board state before making changes (for ko detection)
        self.previous_board_state = self.copy_board_state()

        # Place stone temporarily
        self.board[row][col] = self.current_player

        # Remove captures of adjacent opponent groups
        opponent = 'white' if self.current_player == 'black' else 'black'
        captured_stones = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if self.board[nr][nc] == opponent:
                    if not self.has_liberties(nr, nc, opponent):
                        group = self.get_group(nr, nc, opponent)
                        for gr, gc in group:
                            self.board[gr][gc] = None
                            captured_stones.append((gr, gc))

        captured_count = len(captured_stones)

        # Check suicide rule: if the placed stone (or its group) has no liberties and nothing was captured -> illegal
        if not self.has_liberties(row, col, self.current_player) and captured_count == 0:
            self.board[row][col] = None
            return False

        # Ko detection: handle the common simple-ko case
        # If exactly one stone was captured, check whether this move produced a board identical to the previous board
        # OR check placed group's liberties as an approximation of classic ko detection.
        if captured_count == 1:
            captured_pos = captured_stones[0]

            # Get the group of the stone we just placed
            placed_group = self.get_group(row, col, self.current_player)
            liberties = self.get_group_liberties(placed_group)

            # It's a ko if:
            # 1. Exactly 1 stone was captured
            # 2. The capturing group has exactly 1 liberty
            # 3. That liberty is where the captured stone was
            if len(liberties) == 1 and captured_pos in liberties:
                self.ko_point = captured_pos
                print(f"🔴 Ko situation detected - {captured_pos} is now forbidden")
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Update captures count
        if self.current_player == 'black':
            self.black_captures += captured_count
        else:
            self.white_captures += captured_count

        # Update state
        self.last_move = (row, col)
        self.move_history.append((row, col, self.current_player))
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        self.consecutive_passes = 0
        return True

    def pass_turn(self):
        """Pass the current turn and clear ko point"""
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        self.last_move = None
        self.consecutive_passes += 1
        self.move_history.append(('pass', None, self.current_player))
        self.ko_point = None  # Pass clears ko restriction
        print("⏭️ Pass - ko point cleared")

    def is_game_over(self):
        """Check if game is over (two passes)"""
        return self.consecutive_passes >= 2

    def reset(self):
        """Reset the game"""
        self.__init__()


def draw_menu(screen, buttons, mouse_pos):
    """Draw the main menu"""
    screen.fill(MENU_BG)
    
    # Title with shadow effect
    title_text = "圍碁"  # Go in Chinese/Japanese characters
    title_shadow = FONT_TITLE.render(title_text, True, (80, 50, 20))
    title = FONT_TITLE.render(title_text, True, TITLE_COLOR)
    
    title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
    shadow_rect = title_shadow.get_rect(center=(WINDOW_WIDTH // 2 + 4, 154))
    
    screen.blit(title_shadow, shadow_rect)
    screen.blit(title, title_rect)
    
    # Subtitle
    subtitle = FONT_SUBTITLE.render("The Game of Go", True, TEXT_COLOR)
    subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 230))
    screen.blit(subtitle, subtitle_rect)
    
    # Description
    desc_lines = [
        "An ancient strategy board game",
        "Surround territory to win",
        "Simple rules, infinite depth"
    ]
    y_offset = 300
    for line in desc_lines:
        desc = FONT_SMALL.render(line, True, (120, 80, 40))
        desc_rect = desc.get_rect(center=(WINDOW_WIDTH // 2, y_offset))
        screen.blit(desc, desc_rect)
        y_offset += 35
    
    # Buttons
    for button in buttons:
        color = BUTTON_HOVER if button['rect'].collidepoint(mouse_pos) else BUTTON_COLOR
        
        # Shadow
        shadow = button['rect'].copy()
        shadow.x += 4
        shadow.y += 4
        pygame.draw.rect(screen, (160, 120, 30), shadow, border_radius=15)
        
        # Button
        pygame.draw.rect(screen, color, button['rect'], border_radius=15)
        pygame.draw.rect(screen, TEXT_COLOR, button['rect'], 4, border_radius=15)
        
        # Text
        text = FONT_LARGE.render(button['text'], True, TEXT_COLOR)
        text_rect = text.get_rect(center=button['rect'].center)
        screen.blit(text, text_rect)


def draw_board(screen, game, hover_pos, ai_thinking=False, ai_top_moves=None, show_ai_choices=False):
    """Draw the Go board with optional AI choice highlights"""
    screen.fill(BG_COLOR)
    
    # Board shadow
    shadow_rect = pygame.Rect(
        BOARD_OFFSET_X + 5,
        BOARD_OFFSET_Y + 5,
        BOARD_SIZE_PX,
        BOARD_SIZE_PX
    )
    pygame.draw.rect(screen, (180, 140, 50), shadow_rect)
    
    # Board
    board_rect = pygame.Rect(
        BOARD_OFFSET_X,
        BOARD_OFFSET_Y,
        BOARD_SIZE_PX,
        BOARD_SIZE_PX
    )
    pygame.draw.rect(screen, BOARD_COLOR, board_rect)
    pygame.draw.rect(screen, (160, 120, 30), board_rect, 4)
    
    # Grid lines
    for i in range(BOARD_SIZE):
        x, y = game.get_grid_pos(i, 0)
        x2, y2 = game.get_grid_pos(i, BOARD_SIZE - 1)
        pygame.draw.line(screen, LINE_COLOR, (x, y), (x2, y2), 2)
        
        x, y = game.get_grid_pos(0, i)
        x2, y2 = game.get_grid_pos(BOARD_SIZE - 1, i)
        pygame.draw.line(screen, LINE_COLOR, (x, y), (x2, y2), 2)
    
    # Star points
    star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
    for row, col in star_points:
        x, y = game.get_grid_pos(row, col)
        pygame.draw.circle(screen, LINE_COLOR, (x, y), 5)
    
    # AI top choices highlight (when enabled)
    if show_ai_choices and ai_top_moves:
        for i, ((row, col), prob) in enumerate(ai_top_moves):
            x, y = game.get_grid_pos(row, col)
            
            # Size decreases with rank
            size = int(STONE_RADIUS * (1.2 - i * 0.15))
            alpha = int(180 - i * 30)
            
            # Green highlight
            s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*AI_CHOICE_COLOR[:3], alpha), (size, size), size)
            screen.blit(s, (x - size, y - size))
            
            # Rank number
            rank_text = FONT_SMALL.render(str(i + 1), True, (0, 100, 0))
            rank_rect = rank_text.get_rect(center=(x, y))
            screen.blit(rank_text, rank_rect)
    
    # Hover effect
    if hover_pos and not ai_thinking:
        row, col = hover_pos
        if game.board[row][col] is None:
            x, y = game.get_grid_pos(row, col)
            s = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, HOVER_COLOR, (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
            screen.blit(s, (x - STONE_RADIUS, y - STONE_RADIUS))
    
    # Stones
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if game.board[row][col] is not None:
                x, y = game.get_grid_pos(row, col)
                color = BLACK_STONE if game.board[row][col] == 'black' else WHITE_STONE
                
                # Shadow
                pygame.draw.circle(screen, (0, 0, 0, 50), (x + 2, y + 2), STONE_RADIUS)
                
                # Stone
                pygame.draw.circle(screen, color, (x, y), STONE_RADIUS)
                
                if game.board[row][col] == 'white':
                    pygame.draw.circle(screen, (180, 180, 180), (x, y), STONE_RADIUS, 2)
                
                # Last move marker
                if game.last_move == (row, col):
                    marker_color = WHITE_STONE if game.board[row][col] == 'black' else BLACK_STONE
                    pygame.draw.circle(screen, marker_color, (x, y), 6)


def draw_game_ui(screen, game, buttons, mouse_pos, ai_thinking=False, game_mode='pvp', ai_player=None):
    """Draw game UI elements"""
    # Title (smaller, at top)
    title = FONT_LARGE.render("Go (囲碁)", True, TEXT_COLOR)
    screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
    
    # Current player/status
    status_y = 70
    if game.is_game_over():
        status_text = FONT_MEDIUM.render("Game Over - Two Passes", True, (200, 50, 50))
    elif ai_thinking:
        status_text = FONT_MEDIUM.render("🤖 AI is thinking...", True, AI_THINKING_COLOR)
    else:
        player_name = game.current_player.capitalize()
        if game_mode == 'pve' and ai_player and game.current_player == ai_player:
            player_name += " (AI)"
        elif game_mode == 'pve':
            player_name += " (You)"
        status_text = FONT_MEDIUM.render(f"Turn: {player_name}", True, TEXT_COLOR)
    
    screen.blit(status_text, (WINDOW_WIDTH // 2 - status_text.get_width() // 2, status_y))
    
    # Move counter
    move_count = FONT_SMALL.render(f"Move: {len(game.move_history)}", True, (120, 80, 40))
    screen.blit(move_count, (WINDOW_WIDTH // 2 - move_count.get_width() // 2, 105))
    
    # Black player info (left)
    left_x = 60
    info_y = WINDOW_HEIGHT // 2 - 80
    
    pygame.draw.circle(screen, (50, 50, 50), (left_x + 32, info_y + 2), 32)
    pygame.draw.circle(screen, BLACK_STONE, (left_x + 30, info_y), 30)
    
    black_label = "Black"
    if game_mode == 'pve' and ai_player == 'black':
        black_label += " (AI)"
    black_text = FONT_MEDIUM.render(black_label, True, TEXT_COLOR)
    screen.blit(black_text, (left_x - 10, info_y + 50))
    
    black_cap = FONT_SMALL.render("Captures", True, (140, 100, 50))
    screen.blit(black_cap, (left_x - 15, info_y + 85))
    
    black_num = FONT_LARGE.render(str(game.black_captures), True, TEXT_COLOR)
    screen.blit(black_num, (left_x + 10, info_y + 115))
    
    # White player info (right)
    right_x = WINDOW_WIDTH - 140
    
    pygame.draw.circle(screen, (200, 200, 200), (right_x + 32, info_y + 2), 32)
    pygame.draw.circle(screen, WHITE_STONE, (right_x + 30, info_y), 30)
    pygame.draw.circle(screen, (180, 180, 180), (right_x + 30, info_y), 30, 3)
    
    white_label = "White"
    if game_mode == 'pve' and ai_player == 'white':
        white_label += " (AI)"
    white_text = FONT_MEDIUM.render(white_label, True, TEXT_COLOR)
    screen.blit(white_text, (right_x - 10, info_y + 50))
    
    white_cap = FONT_SMALL.render("Captures", True, (140, 100, 50))
    screen.blit(white_cap, (right_x - 15, info_y + 85))
    
    white_num = FONT_LARGE.render(str(game.white_captures), True, TEXT_COLOR)
    screen.blit(white_num, (right_x + 10, info_y + 115))
    
    # Buttons
    for button in buttons:
        color = BUTTON_HOVER if button['rect'].collidepoint(mouse_pos) else BUTTON_COLOR
        
        shadow = button['rect'].copy()
        shadow.x += 3
        shadow.y += 3
        pygame.draw.rect(screen, (160, 120, 30), shadow, border_radius=10)
        
        pygame.draw.rect(screen, color, button['rect'], border_radius=10)
        pygame.draw.rect(screen, TEXT_COLOR, button['rect'], 3, border_radius=10)
        
        text = FONT_MEDIUM.render(button['text'], True, TEXT_COLOR)
        text_rect = text.get_rect(center=button['rect'].center)
        screen.blit(text, text_rect)


def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Go (囲碁) - The Ancient Game")
    clock = pygame.time.Clock()
    
    # Game state
    game_state = GameState.MENU
    game = GoBoard()
    ai = None
    game_mode = None  # 'pvp' or 'pve'
    ai_player = None  # 'black' or 'white'
    
    # Menu buttons
    menu_buttons = [
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 450, 300, 60),
            'text': 'Player vs Player',
            'action': 'pvp'
        },
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 530, 300, 60),
            'text': 'Play vs AI (You: Black)',
            'action': 'pve_black'
        },
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 610, 300, 60),
            'text': 'Play vs AI (You: White)',
            'action': 'pve_white'
        },
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 690, 300, 60),
            'text': 'Exit',
            'action': 'exit'
        }
    ]
    
    # Game buttons
    button_y = BOARD_OFFSET_Y + BOARD_SIZE_PX + 30
    game_buttons = [
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 - 220, button_y, 100, 45),
            'text': 'Pass',
            'action': 'pass'
        },
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 - 100, button_y, 100, 45),
            'text': 'Undo',
            'action': 'undo'
        },
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 + 20, button_y, 100, 45),
            'text': 'Menu',
            'action': 'menu'
        },
        {
            'rect': pygame.Rect(WINDOW_WIDTH // 2 + 140, button_y, 100, 45),
            'text': 'Reset',
            'action': 'reset'
        }
    ]
    
    # AI visualization settings
    show_ai_thinking = True
    ai_thinking = True
    ai_move_timer = 3000
    ai_delay = 1200  # ms delay for AI visualization
    ai_display_choices_timer = 3000
    ai_display_duration = 3000  # ms to show AI choices
    show_ai_choices = True
    
    hover_pos = None
    running = True
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        dt = clock.tick(60)
        
        # === MENU STATE ===
        if game_state == GameState.MENU:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in menu_buttons:
                        if button['rect'].collidepoint(mouse_pos):
                            action = button['action']
                            
                            if action == 'exit':
                                running = False
                            elif action == 'pvp':
                                game_mode = 'pvp'
                                game.reset()
                                game_state = GameState.PLAYING
                            elif action == 'pve_black':
                                game_mode = 'pve'
                                ai_player = 'white'
                                game.reset()
                                ai = AIPlayer()
                                game_state = GameState.PLAYING
                            elif action == 'pve_white':
                                game_mode = 'pve'
                                ai_player = 'black'
                                game.reset()
                                ai = AIPlayer()
                                game_state = GameState.PLAYING
                                ai_thinking = True
                                ai_move_timer = pygame.time.get_ticks() + ai_delay
            
            draw_menu(screen, menu_buttons, mouse_pos)
        
        # === PLAYING STATE ===
        elif game_state == GameState.PLAYING:
            # Handle AI turn with visualization
            if game_mode == 'pve' and game.current_player == ai_player and not ai_thinking:
                ai_thinking = True
                show_ai_thinking = True
                ai_move_timer = pygame.time.get_ticks() + ai_delay
            
            # Show AI choices for a duration
            if show_ai_choices:
                if pygame.time.get_ticks() >= ai_display_choices_timer:
                    show_ai_choices = False
            
            # Execute AI move after delay
            if ai_thinking and pygame.time.get_ticks() >= ai_move_timer:
                if not show_ai_choices:
                    # Get AI move and show choices
                    move = ai.get_move(game)
                    show_ai_choices = True
                    ai_display_choices_timer = pygame.time.get_ticks() + ai_display_duration
                else:
                    # Execute the move
                    move = ai.get_move(game)  # Re-get to execute
                    if move:
                        row, col = move
                        game.place_stone(row, col)
                    else:
                        game.pass_turn()
                    ai_thinking = False
                    show_ai_thinking = False
                    show_ai_choices = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not ai_thinking:
                    if event.button == 1:
                        # Check buttons
                        button_clicked = False
                        for button in game_buttons:
                            if button['rect'].collidepoint(mouse_pos):
                                action = button['action']
                                button_clicked = True
                                
                                if action == 'pass':
                                    game.pass_turn()
                                elif action == 'undo':
                                    # Simple undo (can be enhanced)
                                    if len(game.move_history) > 0:
                                        print("⚠️ Undo not fully implemented")
                                elif action == 'menu':
                                    game_state = GameState.MENU
                                    game.reset()
                                    ai_thinking = False
                                    show_ai_choices = False
                                elif action == 'reset':
                                    game.reset()
                                    if game_mode == 'pve' and ai_player == 'black':
                                        ai_thinking = True
                                        ai_move_timer = pygame.time.get_ticks() + ai_delay
                                break
                        
                        # Place stone (only for human turn in PvE or anytime in PvP)
                        if not button_clicked:
                            if game_mode == 'pvp' or game.current_player != ai_player:
                                row, col = game.get_board_pos(mouse_pos[0], mouse_pos[1])
                                if row is not None:
                                    game.place_stone(row, col)
                
                elif event.type == pygame.MOUSEMOTION:
                    row, col = game.get_board_pos(mouse_pos[0], mouse_pos[1])
                    hover_pos = (row, col) if row is not None else None
            
            # Draw game
            ai_top_moves = ai.top_moves if ai and show_ai_choices else None
            draw_board(screen, game, hover_pos, ai_thinking, ai_top_moves, show_ai_choices)
            draw_game_ui(screen, game, game_buttons, mouse_pos, show_ai_thinking, game_mode, ai_player)
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()