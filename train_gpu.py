"""
GPU-MAXIMIZED Training with RESIGNATION LOGIC + CATASTROPHIC FORGETTING FIX

CRITICAL FIXES:
1. 50/50 expert/self-play mix (not 80/20) for balance
2. Adaptive learning rate schedule (reduces at iterations 11 and 26)
3. Separate expert buffer for efficient sampling
4. Loss tracking across iterations to detect degradation
5. More conservative resignation thresholds
6. Detailed loss logging (policy + value separate)
7. KEPT: KataGo integration for external opponent training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

import numpy as np
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from collections import deque

from model import GoNet, board_to_input
from board_fast import FastGoBoard
from mcts import BatchedMCTS, select_move
from katago import KataGoOpponent


###############################################################
# GAME GENERATION WITH RESIGNATION
###############################################################

def play_self_play_game_with_resign(model, device, mcts_sims=400, mcts_batch_size=128, 
                                    resign_threshold=-0.9, resign_enabled=True):
    """
    Self-play game with resignation logic
    
    Returns:
        data: Training samples
        winner: 1 (black), -1 (white), 0 (draw), 'resign_black', 'resign_white'
        moves: Number of moves before game ended
    """
    board = FastGoBoard(19)
    mcts = BatchedMCTS(
        model, 
        num_simulations=mcts_sims, 
        temperature=1.0, 
        device=device,
        batch_size=mcts_batch_size,
        resign_threshold=resign_threshold,
        resign_enabled=resign_enabled
    )
    
    # CRITICAL FIX: Reset resignation tracking for new game
    mcts.reset_resignation_tracking()
    
    data = []
    player = 1  # Black starts
    move_num = 0
    resigned = False
    
    # CRITICAL: Hard cap to prevent runaway games even if resignation fails
    MAX_MOVES = 300  # Force end after 300 moves
    
    while not board.is_terminal() and move_num < MAX_MOVES:
        move_num += 1
        mcts.temperature = 1.0 if move_num <= 30 else 0.3
        
        # Get move with resignation check
        policy_dict, should_resign, root_value = mcts.search(board, player, add_dirichlet_noise=True)
        
        # Check resignation
        if should_resign:
            resigned = True
            winner = -player  # Opponent wins
            break
        
        # Store training data
        state = board_to_input(board.board, player, board.get_history())
        policy = np.zeros(362, np.float32)
        for mv, p in policy_dict.items():
            if mv is None or mv == (-1, -1):
                policy[361] = p
            else:
                policy[mv[0] * 19 + mv[1]] = p
        
        data.append((state, policy, player))
        
        # Make move
        move = select_move(policy_dict)
        board.play_move(None if move in [None, (-1, -1)] else move, player)
        mcts.update_root(move)
        player = -player
    
    # Determine winner
    if not resigned:
        winner = board.get_winner()
    
    # Convert to training format
    output = []
    for s, p, pl in data:
        if winner in [0, None]:
            value = 0
        else:
            value = 1 if winner == pl else -1
        output.append((s, p, value))
    
    return output, winner, move_num


def play_game_vs_katago_with_resign(model, katago, device, model_color='black', 
                                   mcts_sims=400, mcts_batch_size=128,
                                   resign_threshold=-0.9):
    """
    Play against KataGo with resignation
    
    Returns:
        data: Training samples
        winner: Game result
        moves: Number of moves
    """
    board = FastGoBoard(19)
    katago.new_game()
    
    mcts = BatchedMCTS(
        model, 
        num_simulations=mcts_sims, 
        temperature=1.0, 
        device=device,
        batch_size=mcts_batch_size,
        resign_threshold=resign_threshold,
        resign_enabled=True
    )
    
    # CRITICAL FIX: Reset resignation tracking for new game
    mcts.reset_resignation_tracking()
    
    data = []
    current_player = 1  # Black
    move_num = 0
    model_player = 1 if model_color == 'black' else -1
    resigned = False
    
    MAX_MOVES = 300  # Force end after 300 moves
    
    while not board.is_terminal() and move_num < MAX_MOVES:
        move_num += 1
        mcts.temperature = 1.0 if move_num <= 30 else 0.3
        
        if current_player == model_player:
            # Your model's turn
            policy_dict, should_resign, root_value = mcts.search(board, current_player, add_dirichlet_noise=True)
            
            # Check resignation
            if should_resign:
                resigned = True
                winner = -model_player  # KataGo wins
                break
            
            # Store data
            state = board_to_input(board.board, current_player, board.get_history())
            policy = np.zeros(362, np.float32)
            for mv, p in policy_dict.items():
                if mv is None or mv == (-1, -1):
                    policy[361] = p
                else:
                    policy[mv[0] * 19 + mv[1]] = p
            
            data.append((state, policy, current_player))
            
            # Make move
            move = select_move(policy_dict)
            
            if move is None or move == (-1, -1):
                board.play_move(None, current_player)
                katago.play_move(None, 'black' if current_player == 1 else 'white')
            else:
                board.play_move(move, current_player)
                katago.play_move(move, 'black' if current_player == 1 else 'white')
            
            mcts.update_root(move)
            
        else:
            # KataGo's turn
            kata_color = 'black' if current_player == 1 else 'white'
            move = katago.get_move(kata_color)
            
            if move == "RESIGN":
                winner = model_player
                break
            
            if move is None:
                board.play_move(None, current_player)
            else:
                board.play_move(move, current_player)
        
        current_player = -current_player
    
    # Determine winner
    if not resigned and board.is_terminal():
        winner = board.get_winner()
    elif not resigned:
        winner = katago.get_winner()
    
    # Convert to training format
    output = []
    for s, p, pl in data:
        if winner in [0, None]:
            value = 0
        else:
            value = 1 if winner == pl else -1
        output.append((s, p, value))
    
    return output, winner, move_num


###############################################################
# REPLAY BUFFER & DATASET
###############################################################

class ReplayBuffer:
    def __init__(self, max_size=30000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, s, p, v):
        self.buffer.append((s, p, v))
    
    def sample(self, size):
        idx = np.random.choice(len(self.buffer), min(size, len(self.buffer)), replace=False)
        samples = [self.buffer[i] for i in idx]
        s, p, v = zip(*samples)
        return np.array(s), np.array(p), np.array(v)
    
    def __len__(self):
        return len(self.buffer)


class GoDataset(Dataset):
    def __init__(self, S, P, V):
        self.S, self.P, self.V = S, P, V
    
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, i):
        return (
            torch.FloatTensor(self.S[i]),
            torch.FloatTensor(self.P[i]),
            torch.FloatTensor([self.V[i]])
        )


class GPUDataset(Dataset):
    def __init__(self, data_file, device, max_samples=None):
        print(f"[DATA] Loading dataset: {data_file}")
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"[DATA] Loaded {len(data):,} samples")
        
        self.states = torch.stack([torch.FloatTensor(s) for s,_,_ in data])
        self.policies = torch.stack([torch.FloatTensor(p) for _,p,_ in data])
        self.values = torch.FloatTensor([[v] for _,_,v in data])
        
        del data
        
        cpu_gb = (self.states.element_size() * self.states.nelement() + 
                  self.policies.element_size() * self.policies.nelement() +
                  self.values.element_size() * self.values.nelement()) / (1024**3)
        
        print(f"[DATA] ✓ Ready ({cpu_gb:.2f} GB in CPU RAM)")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


###############################################################
# TRAINING FUNCTIONS
###############################################################

def train_epoch_gpu(model, dataset, batch_size, opt, scaler, device, accumulation_steps=4):
    """Training with gradient accumulation"""
    model.train()
    
    physical_batch_size = batch_size // accumulation_steps
    
    loader = DataLoader(
        dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    total_loss = 0
    total_policy = 0
    total_value = 0
    
    opt.zero_grad(set_to_none=True)
    
    for batch_idx, (states, policies, values) in enumerate(loader):
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        with autocast('cuda'):
            pol_logits, val_pred = model(states)
            
            loss_policy = -torch.mean(
                torch.sum(policies * torch.log_softmax(pol_logits, dim=1), dim=1)
            )
            loss_value = torch.mean((values - val_pred) ** 2)
            
            loss = (loss_policy + loss_value) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps
        total_policy += loss_policy.item()
        total_value += loss_value.item()
    
    return total_loss / len(loader), total_policy / len(loader), total_value / len(loader)


def train_epoch_standard(model, loader, opt, scaler, device, value_weight=1.0):
    """Standard DataLoader training with adjustable value loss weight"""
    model.train()
    
    total_loss = 0
    total_policy = 0
    total_value = 0
    
    for states, policies, values in loader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            pol_logits, val_pred = model(states)
            
            loss_policy = -torch.mean(
                torch.sum(policies * torch.log_softmax(pol_logits, dim=1), dim=1)
            )
            loss_value = torch.mean((values - val_pred) ** 2)
            # Apply value weight to increase focus on value head
            loss = loss_policy + (value_weight * loss_value)
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        total_loss += loss.item()
        total_policy += loss_policy.item()
        total_value += loss_value.item()
    
    n = len(loader)
    return total_loss / n, total_policy / n, total_value / n


def pretrain_on_foxq(model, dataset_path, device, opt, batch_size, epochs):
    """Pre-train on professional games"""
    print("\n" + "="*60)
    print("PHASE 1: SUPERVISED LEARNING")
    print("="*60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    dataset = GPUDataset(dataset_path, device)
    scaler = GradScaler('cuda')
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n[{timestamp}] Starting Epoch {epoch+1}/{epochs}...")
        
        loss, policy_loss, value_loss = train_epoch_gpu(
            model, dataset, batch_size, opt, scaler, device, accumulation_steps=4
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start
        timestamp_end = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{timestamp_end}] Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Time: {epoch_time:.1f}s")
    
    print("\n✓ Pre-training complete!")
    return model


###############################################################
# MAIN TRAINING
###############################################################

def train_from_scratch(
    foxq_dataset_path="foxq_elite_7d_plus.pkl",
    pretrain_epochs=15,
    iterations=100,
    games_per_iter=20,
    mcts_sims=400,
    rl_epochs=12,
    batch_size=512,
    learning_rate=0.0001,
    save_every=3,
    checkpoint_dir="checkpoints",
    # Resignation settings
    initial_resign_threshold=-0.90,
    final_resign_threshold=-0.80,
    resign_enabled=True,
):
    """
    Train from scratch with fixed resignation and learning
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("\n🚨 CUDA not available! Training on CPU will be VERY slow.")
        return
    
    print("\n" + "="*70)
    print("TRAINING FROM SCRATCH - FIXED VERSION")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")
    print(f"MCTS sims: {mcts_sims}")
    print(f"Resignation: {resign_enabled}")
    print(f"Expert/Self-play mix: ADAPTIVE (70%→60%→50% expert)")
    print(f"LR schedule: {learning_rate} → {learning_rate*0.5} → {learning_rate*0.25}")
    print("="*70)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model = GoNet.create_small().to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda')
    buf = ReplayBuffer(max_size=30000)
    writer = SummaryWriter("runs/training_fixed")
    
    # Check for existing checkpoints
    pretrained_path = Path(checkpoint_dir) / "model_pretrained.pt"
    final_path = Path(checkpoint_dir) / "model_final.pt"
    
    start_iteration = 1
    pretrained = False
    
    # Try to load existing checkpoint (priority: final > pretrained)
    if final_path.exists():
        print(f"\n[RESUME] Loading checkpoint: {final_path}")
        ckpt = torch.load(final_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_iteration = ckpt.get('iteration', 0) + 1
        pretrained = True
        print(f"[RESUME] Resuming from iteration {start_iteration}")
    elif pretrained_path.exists():
        print(f"\n[RESUME] Loading pretrained model: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        pretrained = True
        print(f"[RESUME] Loaded pretrained model, starting RL from iteration 1")
    
    # PHASE 1: Pre-training (only if no pretrained model exists)
    dataset_path = Path(foxq_dataset_path)
    if not pretrained and dataset_path.exists():
        print(f"\n[FOXQ] Found dataset: {dataset_path}")
        model = pretrain_on_foxq(model, str(dataset_path), device, opt, batch_size, pretrain_epochs)
        
        torch.save({
            'iteration': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'pretrained': True,
        }, pretrained_path)
        pretrained = True
    elif not pretrained:
        print("\n⚠️  WARNING: No pretrained model found and no dataset available!")
        print("Training will start from random initialization.")
    
    # PHASE 2: Reinforcement Learning
    print("\n" + "="*60)
    print("PHASE 2: REINFORCEMENT LEARNING (FIXED)")
    print("="*60)
    
    # CRITICAL FIX: Load expert data into separate buffer
    expert_buffer = ReplayBuffer(max_size=50000)
    if dataset_path.exists():
        print(f"\n[EXPERT] Loading expert data for mixing...")
        with open(str(dataset_path), 'rb') as f:
            expert_data = pickle.load(f)
        # Pre-load expert data into buffer
        print(f"[EXPERT] Pre-loading {min(50000, len(expert_data)):,} expert samples...")
        sample_indices = np.random.choice(len(expert_data), min(50000, len(expert_data)), replace=False)
        for idx in sample_indices:
            s, p, v = expert_data[idx]
            expert_buffer.add(s, p, v)
        print(f"[EXPERT] Loaded {len(expert_buffer):,} expert samples")
        print(f"[EXPERT] Strategy: ADAPTIVE mix to prevent value head collapse")
        print(f"[EXPERT]   Iterations 1-10:  70% expert (stabilization)")
        print(f"[EXPERT]   Iterations 11-25: 60% expert (transition)")
        print(f"[EXPERT]   Iterations 26+:   50% expert (balanced)")
        print(f"[EXPERT] LR schedule: reduces at iter 11 and 26")
    
    step = 0
    game_lengths = deque(maxlen=100)
    resignations = deque(maxlen=100)
    iteration_losses = deque(maxlen=10)
    best_loss = float('inf')
    
    for it in range(start_iteration, iterations + 1):
        iter_start = time.time()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'='*60}")
        print(f"[{timestamp}] ITERATION {it}/{iterations}")
        print(f"{'='*60}")
        
        # CRITICAL FIX: Adaptive learning rate
        if it == 11:
            for param_group in opt.param_groups:
                param_group['lr'] = learning_rate * 0.5
            print(f"\n[LR] Reduced learning rate to {learning_rate * 0.5:.6f}")
        elif it == 26:
            for param_group in opt.param_groups:
                param_group['lr'] = learning_rate * 0.25
            print(f"\n[LR] Reduced learning rate to {learning_rate * 0.25:.6f}")
        
        # Gradually make resignation more aggressive
        progress = it / iterations
        current_threshold = initial_resign_threshold + (final_resign_threshold - initial_resign_threshold) * progress
        
        # Generate games
        model.eval()
        games_start = time.time()
        print(f"\n[GAMES] Generating {games_per_iter} games (resign threshold: {current_threshold:.2f})...")
        
        iter_resignations = 0
        
        for g in range(1, games_per_iter + 1):
            game_data, winner, moves = play_self_play_game_with_resign(
                model, device, mcts_sims, mcts_batch_size=128,
                resign_threshold=current_threshold,
                resign_enabled=resign_enabled
            )
            
            # Track stats
            game_lengths.append(moves)
            if winner is not None and abs(winner) == 1 and moves < 250:
                iter_resignations += 1
                resignations.append(1)
            else:
                resignations.append(0)
            
            # Add to buffer
            for s, p, v in game_data:
                buf.add(s, p, v)
            
            if g % 5 == 0:
                elapsed = time.time() - games_start
                avg_time = elapsed / g
                avg_length = sum(list(game_lengths)[-g:]) / min(g, len(game_lengths))
                print(f"  Game {g}/{games_per_iter}: {moves} moves | {elapsed:.1f}s elapsed ({avg_time:.1f}s/game, avg: {avg_length:.0f})")
        
        games_time = time.time() - games_start
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        resign_rate = sum(resignations) / len(resignations) if resignations else 0
        
        print(f"\n[STATS] Avg game length: {avg_game_length:.0f} moves | Resignation rate: {resign_rate:.1%}")
        print(f"[BUFFER] Total samples: {len(buf):,}")
        print(f"[TIMING] Game generation: {games_time:.1f}s ({games_time/games_per_iter:.1f}s per game)")
        
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training
        if len(buf) >= batch_size:
            model.train()
            timestamp_train = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp_train}] [TRAIN] Training for {rl_epochs} epochs...")
            
            # ADAPTIVE MIX: Use more expert data early to stabilize value head
            # Iterations 1-10: 70% expert (value head stabilization)
            # Iterations 11-25: 60% expert (transition)
            # Iterations 26+: 50% expert (balanced learning)
            
            if it <= 10:
                expert_ratio = 0.70  # 70% expert early on
            elif it <= 25:
                expert_ratio = 0.60  # 60% expert in middle
            else:
                expert_ratio = 0.50  # 50% expert later
            
            # Calculate sample sizes based on adaptive ratio
            n_self_play = min(batch_size * 3, len(buf))
            S, P, V = buf.sample(n_self_play)
            
            # Mix in expert data with adaptive ratio
            if len(expert_buffer) > 0:
                # Calculate expert samples needed for desired ratio
                # If we want 70% expert, and we have X self-play samples:
                # expert / (expert + self_play) = 0.70
                # expert = 0.70 * (expert + self_play)
                # expert = 0.70 * expert + 0.70 * self_play
                # 0.30 * expert = 0.70 * self_play
                # expert = (0.70 / 0.30) * self_play
                n_expert = int(n_self_play * (expert_ratio / (1 - expert_ratio)))
                n_expert = min(n_expert, len(expert_buffer))
                
                S_expert, P_expert, V_expert = expert_buffer.sample(n_expert)
                
                S = np.concatenate([S, S_expert], axis=0)
                P = np.concatenate([P, P_expert], axis=0)
                V = np.concatenate([V, V_expert], axis=0)
                
                actual_expert_pct = n_expert / len(S) * 100
                actual_self_play_pct = n_self_play / len(S) * 100
                print(f"[MIX] {n_self_play:,} self-play ({actual_self_play_pct:.1f}%) + {n_expert:,} expert ({actual_expert_pct:.1f}%) = {len(S):,} total")
            
            # Shuffle
            perm = np.random.permutation(len(S))
            S = S[perm]
            P = P[perm]
            V = V[perm]
            
            train_ds = GoDataset(S, P, V)
            loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            
            epoch_losses = []
            for e in range(rl_epochs):
                epoch_start = time.time()
                
                # CRITICAL: Increase value loss weight when value head is collapsing
                # When value loss < 0.1, increase weight to force learning
                avg_value_loss = np.mean([iteration_losses[i] for i in range(max(0, len(iteration_losses)-3), len(iteration_losses))]) if iteration_losses else 1.0
                value_weight = 2.0 if avg_value_loss < 0.15 else 1.0
                
                loss, policy_loss, value_loss = train_epoch_standard(model, loader, opt, scaler, device, value_weight=value_weight)
                epoch_time = time.time() - epoch_start
                epoch_losses.append(loss)
                
                writer.add_scalar("loss/total", loss, step)
                writer.add_scalar("loss/policy", policy_loss, step)
                writer.add_scalar("loss/value", value_loss, step)
                writer.add_scalar("stats/avg_game_length", avg_game_length, step)
                writer.add_scalar("stats/resign_rate", resign_rate, step)
                writer.add_scalar("learning_rate", opt.param_groups[0]['lr'], step)
                writer.add_scalar("value_weight", value_weight, step)
                step += 1
                
                if e == 0 or e == rl_epochs - 1:
                    timestamp_epoch = datetime.now().strftime("%H:%M:%S")
                    weight_str = f" [value_weight: {value_weight:.1f}]" if value_weight != 1.0 else ""
                    print(f"  [{timestamp_epoch}] Epoch {e+1}/{rl_epochs} | Total: {loss:.4f} (policy: {policy_loss:.4f}, value: {value_loss:.4f}){weight_str} | {epoch_time:.1f}s")
            
            # Track improvement
            final_loss = epoch_losses[-1]
            iteration_losses.append(final_loss)
            
            if final_loss < best_loss:
                best_loss = final_loss
                print(f"[TRAIN] ✓ New best loss: {best_loss:.4f}")
            
            # Warning if degrading
            if len(iteration_losses) >= 3:
                recent_trend = iteration_losses[-1] - iteration_losses[-3]
                if recent_trend > 0.1:
                    print(f"[TRAIN] ⚠️  WARNING: Loss increasing (+{recent_trend:.4f} over 3 iters)")
        
        # Save checkpoint
        if it % save_every == 0 or it == iterations:
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'pretrained': True,
                'resign_threshold': current_threshold,
                'best_loss': best_loss,
            }, Path(checkpoint_dir) / "model_final.pt")
            timestamp_save = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp_save}] [SAVE] ✓ Checkpoint: checkpoints/model_final.pt")
            
            if it in [3, 10, 25, 50, 75, 100]:
                milestone_path = Path(checkpoint_dir) / f"model_iter_{it}.pt"
                torch.save({
                    'iteration': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'pretrained': True,
                }, milestone_path)
                print(f"[{timestamp_save}] [SAVE] ✓ Milestone: {milestone_path}")
        
        iter_time = time.time() - iter_start
        remaining = (iterations - it) * iter_time
        
        timestamp_end = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp_end}] [TIME] Iteration: {iter_time/60:.1f} min | Remaining: {remaining/60:.1f} min")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    writer.close()


if __name__ == "__main__":
    train_from_scratch(
        foxq_dataset_path="foxq_elite_7d_plus.pkl",
        pretrain_epochs=25,
        iterations=100,
        games_per_iter=20,
        mcts_sims=400,
        rl_epochs=12,
        batch_size=256,
        learning_rate=0.0001,
        save_every=3,
        checkpoint_dir="checkpoints",
        # CRITICAL: Resignation disabled to prevent value head collapse death spiral
        initial_resign_threshold=-0.90,
        final_resign_threshold=-0.80,
        resign_enabled=False,  # ← CHANGED FROM TRUE TO FALSE
    )