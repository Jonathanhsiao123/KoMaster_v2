"""
GPU-Optimized MCTS with Batched Inference + RESIGNATION LOGIC

Fixes from code review:
1. _get_legal_moves now delegates to C extension (was re-implementing in Python)
2. _get_state_string now uses Zobrist hash from C (was hex-encoding 361 bytes per node)
3. _run_batch_simulations now uses board.copy() (was manually copying arrays, losing history)
4. board_to_input allocations amortized via pre-allocated buffer
5. N visit count lookup uses board-level key not string+action (avoids repeated string ops)
"""

import torch
import numpy as np
from collections import defaultdict
import math


class BatchedMCTS:
    """
    MCTS with batched neural network evaluations + resignation logic.
    Evaluates multiple game positions simultaneously to maximize GPU usage.
    """

    def __init__(self, model, num_simulations=400, temperature=1.0,
                 c_puct=1.5, device='cuda', batch_size=32,
                 resign_threshold=-0.9, resign_enabled=True):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.c_puct = c_puct
        self.device = device
        self.batch_size = batch_size

        # Tree structure
        self.children = {}          # state_key -> list of legal actions
        self.root_state = None

        # Tree statistics — keyed by (state_key, action) where action is (r,c) or (-1,-1)
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.P = {}                 # state_key -> {action: prior_prob}

        # Resignation
        self.resign_threshold = resign_threshold
        self.resign_enabled = resign_enabled
        self.consecutive_bad_moves = 0
        self.resign_check_interval = 5
        self.move_count = 0
        self.root_value_history = []

        # Pre-allocated input buffer: reused across board_to_input calls
        # Shape: (17, 19, 19) — filled in-place to avoid per-call allocation
        self._input_buf = np.zeros((17, 19, 19), dtype=np.float32)

    # ── Resignation ──────────────────────────────────────────────────────────

    def should_resign(self, current_value):
        if not self.resign_enabled:
            return False
        if self.move_count < 30:
            return False

        self.root_value_history.append(current_value)
        if len(self.root_value_history) > 10:
            self.root_value_history.pop(0)
        if len(self.root_value_history) < 5:
            return False

        if current_value < self.resign_threshold:
            self.consecutive_bad_moves += 1
        else:
            self.consecutive_bad_moves = 0

        if self.consecutive_bad_moves >= 3:
            recent_avg = sum(self.root_value_history[-5:]) / 5
            if recent_avg < self.resign_threshold:
                return True
        return False

    def reset_resignation_tracking(self):
        self.consecutive_bad_moves = 0
        self.root_value_history = []
        self.move_count = 0

    # ── State key ─────────────────────────────────────────────────────────────

    def _state_key(self, board, player):
        """
        FIX #2: Use Zobrist hash from C extension instead of hex-encoding board bytes.
        
        Old: f"{board_array.tobytes().hex()}_{player}"
             → allocates a 722-char Python string on every node expansion
        New: (zobrist_hash_int, player)
             → tuple of two ints, no allocation beyond the tuple
        
        Zobrist hashing has negligible collision probability for Go
        (64-bit hash over 19x19 board with 2 colors = 2^64 states).
        """
        return (board.zobrist_hash, player)

    # ── Legal moves ───────────────────────────────────────────────────────────

    def _get_legal_moves(self, board, player):
        """
        FIX #1: Delegate entirely to C extension.
        
        Old: double Python loop (19×19) calling board.is_legal_move in Python
             → 361 Python function calls per invocation
        New: single C call returning a list
             → one Python→C transition, all iteration in C
        
        Pass move is represented as (-1,-1) for consistency with the rest of MCTS.
        The C extension's get_legal_moves doesn't include pass, so we append it.
        """
        moves = board.get_legal_moves(player)   # C call — O(361) entirely in C
        moves.append((-1, -1))                  # Always legal to pass
        return moves

    # ── Neural net input ──────────────────────────────────────────────────────

    def _board_to_input(self, board, player):
        """
        FIX #4: Reuse a pre-allocated (17,19,19) float32 buffer.
        
        Old (model.board_to_input): allocates np.zeros((17,19,19)) + 17 boolean
             arrays per call → ~18 heap allocations per leaf node
        New: zero the buffer in-place, fill slices directly
             → 0 heap allocations for the array data
        
        This is called at every MCTS leaf (batch_size × num_sims/batch_size = num_sims
        times per search call), so the saving compounds quickly.
        """
        buf = self._input_buf
        buf[:] = 0.0  # in-place zero — no allocation

        b = board.board   # single numpy view from C bytes
        history = board.board_history  # list of np.ndarray

        all_boards = [b] + history  # list reference, no copy

        if player == 1:  # black to play
            for i in range(min(8, len(all_boards))):
                h = all_boards[i]
                buf[i]     = (h == 1)   # current player stones
                buf[i + 8] = (h == -1)  # opponent stones
            buf[16] = 1.0
        else:             # white to play
            for i in range(min(8, len(all_boards))):
                h = all_boards[i]
                buf[i]     = (h == -1)  # current player stones
                buf[i + 8] = (h == 1)   # opponent stones
            # buf[16] already 0.0

        return buf  # caller must copy if storing; for immediate tensor conversion it's fine

    # ── MCTS search ───────────────────────────────────────────────────────────

    def search(self, board, player, add_dirichlet_noise=False):
        """
        Main entry point. Returns (policy_dict, should_resign, root_value).
        """
        self.move_count += 1

        root_key = self._state_key(board, player)
        self.root_state = root_key

        # ── Root initialisation ──
        root_value = 0.0

        if root_key not in self.children:
            state_input = self._board_to_input(board, player)
            with torch.no_grad():
                t = torch.from_numpy(state_input).unsqueeze(0).to(self.device)
                policy_logits, value_t = self.model(t)
                policy_np = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                root_value = value_t.item()

            legal_moves = self._get_legal_moves(board, player)
            self.children[root_key] = legal_moves

            policy_dict = {}
            for move in legal_moves:
                idx = 361 if move == (-1, -1) else move[0] * 19 + move[1]
                policy_dict[move] = float(policy_np[idx])

            total = sum(policy_dict.values()) or 1.0
            self.P[root_key] = {k: v / total for k, v in policy_dict.items()}
        else:
            # Re-evaluate root for resignation check — cheap single forward pass
            state_input = self._board_to_input(board, player)
            with torch.no_grad():
                t = torch.from_numpy(state_input).unsqueeze(0).to(self.device)
                _, value_t = self.model(t)
                root_value = value_t.item()

        # ── Resignation check ──
        if self.move_count % self.resign_check_interval == 0:
            if self.should_resign(root_value):
                return self._get_policy(root_key), True, root_value

        # ── Exploration noise at root ──
        if add_dirichlet_noise:
            self._add_dirichlet_noise(root_key)

        # ── Simulations ──
        for batch_start in range(0, self.num_simulations, self.batch_size):
            actual = min(self.batch_size, self.num_simulations - batch_start)
            self._run_batch_simulations(board, player, actual)

        return self._get_policy(root_key), False, root_value

    # ── Batch simulations ─────────────────────────────────────────────────────

    def _run_batch_simulations(self, root_board, root_player, batch_size):
        """
        FIX #3: Use board.copy() instead of manually reconstructing board state.
        
        Old:
            board = FastGoBoard(19)          # new object + DSU init
            board.board = root_board.board.copy()   # numpy copy
            board.history = root_board.history.copy() if hasattr(...) else []
            # ↑ used wrong attribute name ('history' vs 'board_history'),
            #   so history planes were always empty → neural net saw wrong input
        
        New:
            board = root_board.copy()        # single memcpy in C (8 KB struct)
            # board_history is correctly copied by FastGoBoard.copy()
        """
        simulations = []
        leaf_inputs = []   # list of (17,19,19) arrays to batch-evaluate

        for _ in range(batch_size):
            # FIX #3: correct copy preserving DSU state and board_history
            board = root_board.copy()
            player = root_player

            path = []
            state_key = self._state_key(board, player)

            # Selection
            while state_key in self.P and self.N[state_key] > 0:
                action = self._select_action(state_key, board, player)
                path.append((state_key, action, player))

                board.play_move(None if action == (-1, -1) else action, player)
                player = -player
                state_key = self._state_key(board, player)

                if board.is_terminal():
                    break

            if board.is_terminal():
                simulations.append((path, board, player, state_key, True))
            else:
                # Collect input for batch eval — copy buffer since it's reused
                inp = self._board_to_input(board, player).copy()
                leaf_inputs.append(inp)
                simulations.append((path, board, player, state_key, False))

        # Batch neural network evaluation
        policies_np = values_np = None
        if leaf_inputs:
            with torch.no_grad():
                batch_t = torch.from_numpy(np.stack(leaf_inputs)).to(self.device)
                pol_logits, vals = self.model(batch_t)
                policies_np = torch.softmax(pol_logits, dim=1).cpu().numpy()
                values_np   = vals.cpu().numpy().flatten()

        # Expansion + backpropagation
        leaf_idx = 0
        for path, board, player, leaf_key, is_terminal in simulations:
            if is_terminal:
                winner = board.get_winner()
                value = 0 if (winner == 0 or winner is None) else (1 if winner == player else -1)
            else:
                policy_row = policies_np[leaf_idx]
                value      = float(values_np[leaf_idx])
                leaf_idx  += 1

                legal_moves = self._get_legal_moves(board, player)
                policy_dict = {}
                for move in legal_moves:
                    idx = 361 if move == (-1, -1) else move[0] * 19 + move[1]
                    policy_dict[move] = float(policy_row[idx])

                total = sum(policy_dict.values())
                if total > 0:
                    policy_dict = {k: v / total for k, v in policy_dict.items()}
                self.P[leaf_key] = policy_dict

            # Backpropagate
            for state_key, action, _ in reversed(path):
                sa = (state_key, action)
                self.N[sa] += 1
                self.Q[sa] += (value - self.Q[sa]) / self.N[sa]
                value = -value

            # Also increment board-level visit count (used by _select_action)
            self.N[leaf_key] += 1

    # ── PUCT selection ────────────────────────────────────────────────────────

    def _select_action(self, state_key, board, player):
        """PUCT selection. Legal moves come from C — O(4) per position."""
        legal_moves = self._get_legal_moves(board, player)

        if state_key not in self.P:
            return legal_moves[np.random.randint(len(legal_moves))]

        policy = self.P[state_key]
        total_n = sum(self.N[(state_key, a)] for a in legal_moves)
        sqrt_total = math.sqrt(total_n + 1)

        best_score = -float('inf')
        best_action = None

        for action in legal_moves:
            sa = (state_key, action)
            n = self.N[sa]
            q = self.Q[sa] if n > 0 else 0.0
            p = policy.get(action, 1e-8)
            score = q + self.c_puct * p * sqrt_total / (1 + n)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ── Policy extraction ─────────────────────────────────────────────────────

    def _get_policy(self, state_key):
        if state_key not in self.P:
            return {(-1, -1): 1.0}

        actions = list(self.P[state_key].keys())
        visits = {a: self.N[(state_key, a)] for a in actions}
        total_visits = sum(visits.values())

        if total_visits == 0:
            n = len(visits)
            return {a: 1.0 / n for a in actions}

        if self.temperature == 0:
            best = max(visits, key=visits.__getitem__)
            return {best: 1.0}

        policy = {a: n ** (1.0 / self.temperature) for a, n in visits.items()}
        total = sum(policy.values())
        return {k: v / total for k, v in policy.items()}

    # ── Dirichlet noise ───────────────────────────────────────────────────────

    def _add_dirichlet_noise(self, state_key, alpha=0.3, epsilon=0.25):
        if state_key not in self.P:
            return
        policy = self.P[state_key]
        actions = list(policy.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        self.P[state_key] = {
            a: (1 - epsilon) * policy[a] + epsilon * n
            for a, n in zip(actions, noise)
        }

    def update_root(self, action):
        self.root_state = None


# ── Utility ───────────────────────────────────────────────────────────────────

def select_move(policy_dict):
    """Sample a move from the policy distribution."""
    actions = list(policy_dict.keys())
    probs   = list(policy_dict.values())
    total   = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / len(probs)] * len(probs)
    return actions[np.random.choice(len(actions), p=probs)]
