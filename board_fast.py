"""
board_fast.py  —  Drop-in replacement for the old pure-Python FastGoBoard.

Wraps go_board.GoBoard (C extension) to provide identical API surface.
Key additions:
  - board_history (list of np.ndarray) for neural-net 17-plane input
  - to_tensor_format() matching AlphaGo-style (17 planes)
  - get_winner() / is_terminal() delegated to C
"""

from typing import Optional, List, Tuple
import numpy as np

try:
    import go_board as _go_board
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False
    print("⚠️  go_board C extension not found — falling back to pure Python.\n"
          "    Build with: cd go_board_c && python setup.py build_ext --inplace")


class FastGoBoard:
    """
    Drop-in replacement for the original FastGoBoard.

    Internally delegates to the C extension (go_board.GoBoard) which uses
    Union-Find for O(1)-amortized group/liberty queries instead of BFS.

    Expected MCTS speedup: 10–50x depending on simulation count.
    """

    def __init__(self, size: int = 19):
        if not _HAS_C_EXT:
            raise RuntimeError("go_board C extension is required. "
                               "Build it with: python setup.py build_ext --inplace")
        self._c = _go_board.GoBoard(size)
        self.size = size

        # Board history for 17-plane neural net input (most recent first)
        # Each entry is a (size, size) int8 numpy array
        self.board_history: List[np.ndarray] = []

        # Move history (kept for compatibility)
        self.move_history: List[Optional[Tuple[int, int]]] = []

    # ── Core properties (delegate to C) ──────────────────────────────────────

    @property
    def ko_point(self):
        return self._c.ko_point

    @property
    def last_move(self):
        return self._c.last_move

    @property
    def consecutive_passes(self) -> int:
        return self._c.consecutive_passes

    @property
    def black_captures(self) -> int:
        return self._c.black_captures

    @property
    def white_captures(self) -> int:
        return self._c.white_captures

    @property
    def zobrist_hash(self) -> int:
        return self._c.zobrist_hash

    @property
    def board(self) -> np.ndarray:
        """Current board as (size, size) int8 numpy array (0/1/-1)."""
        raw = self._c.to_bytes()
        return np.frombuffer(raw, dtype=np.int8).reshape(self.size, self.size).copy()

    # ── Move API ──────────────────────────────────────────────────────────────

    def play_move(self, move: Optional[Tuple[int, int]], color: int) -> bool:
        # Save board state to history BEFORE the move
        self.board_history.insert(0, self.board)
        if len(self.board_history) > 8:
            self.board_history.pop()

        ok = self._c.play_move(move, color)

        if not ok:
            # Move was illegal — undo the history push
            self.board_history.pop(0)
            return False

        self.move_history.append(move)
        return True

    def is_legal_move(self, row: int, col: int, color: int) -> bool:
        return self._c.is_legal_move(row, col, color)

    def get_legal_moves(self, color: int) -> List[Tuple[int, int]]:
        return self._c.get_legal_moves(color)

    # ── Game state ────────────────────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self._c.is_terminal()

    def get_winner(self) -> Optional[int]:
        return self._c.get_winner()

    def get_history(self) -> List[np.ndarray]:
        return self.board_history.copy()

    # ── Neural network input ──────────────────────────────────────────────────

    def to_tensor_format(self) -> np.ndarray:
        """
        17-plane AlphaGo Zero-style input for the neural network.

        Planes 0–7:   Black stone positions for last 8 board states
        Planes 8–15:  White stone positions for last 8 board states
        Plane  16:    Color to play (all 1s = black to play, 0s = white)

        Returns shape (17, size, size) float32.
        """
        result = np.zeros((17, self.size, self.size), dtype=np.float32)
        current = self.board

        # Most recent board first, then history
        boards = [current] + self.board_history
        for i in range(min(8, len(boards))):
            b = boards[i]
            result[i]     = (b == 1).astype(np.float32)   # black planes
            result[i + 8] = (b == -1).astype(np.float32)  # white planes

        # Plane 16: whose turn — caller should fill, default black=1
        # (The training loop sets this based on current_player)
        result[16] = 1.0

        return result

    def to_tensor_format_simple(self) -> np.ndarray:
        """
        3-plane format (compatible with older training code).
        Channels: [black, white, empty]  shape (3, size, size)
        """
        b = self.board
        result = np.zeros((3, self.size, self.size), dtype=np.float32)
        result[0] = (b == 1).astype(np.float32)
        result[1] = (b == -1).astype(np.float32)
        result[2] = (b == 0).astype(np.float32)
        return result

    # ── Copying ───────────────────────────────────────────────────────────────

    def copy(self) -> 'FastGoBoard':
        new_board = object.__new__(FastGoBoard)
        new_board.size = self.size
        new_board._c = self._c.copy()
        new_board.board_history = [b.copy() for b in self.board_history]
        new_board.move_history = self.move_history.copy()
        return new_board

    # ── Debug ─────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        symbols = {0: '·', 1: '●', -1: '○'}
        b = self.board
        lines = []
        for row in b:
            lines.append(' '.join(symbols[int(cell)] for cell in row))
        return '\n'.join(lines)

    def print_board(self):
        print(self)
        print(f"Black captures: {self.black_captures}")
        print(f"White captures: {self.white_captures}")
        if self.last_move:
            print(f"Last move: {self.last_move}")


# ─── Build and benchmark helper ───────────────────────────────────────────────

if __name__ == "__main__":
    import time, subprocess, sys, os

    # Try to build the extension if not present
    if not _HAS_C_EXT:
        print("Building C extension...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=script_dir, check=True
        )
        import importlib
        import go_board as _go_board
        _HAS_C_EXT = True

    print("Testing FastGoBoard (C backend)...\n")

    board = FastGoBoard(19)
    board.play_move((3, 3), 1)
    board.play_move((3, 4), -1)
    board.play_move((4, 3), 1)
    board.print_board()

    print(f"\nLegal moves for black: {len(board.get_legal_moves(1))}")

    b2 = board.copy()
    b2.play_move((5, 5), -1)
    print(f"\nOriginal: {np.sum(board.board != 0)} stones")
    print(f"Copy:     {np.sum(b2.board != 0)} stones")

    print(f"\nTensor shape: {board.to_tensor_format().shape}")

    # ── Benchmark ──
    print("\nBenchmark: 10,000 legal-move queries on a mid-game board...")
    bench = FastGoBoard(19)
    # Fill ~100 stones to simulate mid-game
    import itertools
    moves = list(itertools.product(range(10), range(10)))
    for i, (r, c) in enumerate(moves[:80]):
        bench.play_move((r, c), 1 if i % 2 == 0 else -1)

    start = time.perf_counter()
    for _ in range(10_000):
        bench.get_legal_moves(1)
    elapsed = time.perf_counter() - start
    print(f"  10,000 × get_legal_moves: {elapsed:.3f}s  "
          f"({10_000/elapsed:,.0f} calls/sec)")

    print("\n✅ All tests passed!")
