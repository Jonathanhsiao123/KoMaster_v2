"""
KataGo opponent wrapper for RL training

SETUP:
1. Download KataGo: https://github.com/lightvector/KataGo/releases
2. Ungzip model: gunzip kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
3. Place in project directory
"""

import subprocess
import re
import numpy as np
from pathlib import Path

class KataGoOpponent:
    """
    Interface to KataGo for playing as opponent
    
    CHANGES:
    - Replaces self-play with play vs KataGo
    - Your model learns from losses/wins against superhuman AI
    - Games end naturally (KataGo knows when to pass)
    """
    
    def __init__(self, 
                 katago_path="./katago.exe",
                 model_path="kata1-b28c512nbt-adam-s11165M-d5387M.bin",
                 config_path="gtp_config.cfg",
                 num_visits=100):  # KataGo search depth
        """
        Initialize KataGo opponent
        
        Args:
            katago_path: Path to KataGo binary
            model_path: Path to KataGo model weights (.bin file, ungzipped)
            config_path: Path to KataGo config
            num_visits: KataGo MCTS visits (100=weak, 400=medium, 1600=strong)
        """
        self.katago_path = katago_path
        self.model_path = model_path
        self.config_path = config_path
        self.num_visits = num_visits
        self.process = None
        
        # Verify files exist
        if not Path(katago_path).exists():
            raise FileNotFoundError(f"KataGo binary not found: {katago_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"KataGo model not found: {model_path}")
        
        self._start_katago()
    
    def _start_katago(self):
        """Start KataGo GTP process"""
        cmd = [
            self.katago_path,
            "gtp",
            "-model", self.model_path,
            "-config", self.config_path,
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Set search limits
        self._send_command(f"kata-set-param maxVisits {self.num_visits}")
        print(f"✓ KataGo started (visits={self.num_visits})")
    
    def _send_command(self, command):
        """Send GTP command to KataGo and get response"""
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        
        # Read response until blank line
        response_lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == "":
                break
            response_lines.append(line)
        
        # Parse response
        if response_lines and response_lines[0].startswith("="):
            # Success
            return " ".join(response_lines[0][1:].strip().split())
        else:
            # Error
            return None
    
    def new_game(self, board_size=19, komi=7.5):
        """Start a new game"""
        self._send_command(f"boardsize {board_size}")
        self._send_command("clear_board")
        self._send_command(f"komi {komi}")
    
    def get_move(self, color="white"):
        """
        Get KataGo's move
        
        Args:
            color: "white" or "black"
        
        Returns:
            (row, col) tuple or None for pass
        """
        response = self._send_command(f"genmove {color}")
        
        if response is None or response.lower() == "pass":
            return None
        
        # Parse GTP coordinate (e.g., "D4")
        if response.lower() == "resign":
            return "RESIGN"
        
        col_letter = response[0].upper()
        row_num = int(response[1:])
        
        # Convert GTP to (row, col)
        col = ord(col_letter) - ord('A')
        if col >= 8:  # Skip 'I' in GTP
            col -= 1
        row = 19 - row_num  # GTP counts from bottom
        
        return (row, col)
    
    def play_move(self, move, color="black"):
        """
        Tell KataGo about a move
        
        Args:
            move: (row, col) tuple or None for pass
            color: "black" or "white"
        """
        if move is None:
            gtp_move = "pass"
        else:
            row, col = move
            # Convert to GTP coordinate
            if col >= 8:
                col += 1  # Skip 'I'
            col_letter = chr(ord('A') + col)
            row_num = 19 - row
            gtp_move = f"{col_letter}{row_num}"
        
        self._send_command(f"play {color} {gtp_move}")
    
    def get_winner(self):
        """
        Get final score
        
        Returns:
            1 for black win, -1 for white win, 0 for draw
        """
        response = self._send_command("final_score")
        
        if response is None:
            return 0
        
        # Parse score (e.g., "B+5.5" or "W+12.0")
        if response.startswith("B"):
            return 1  # Black wins
        elif response.startswith("W"):
            return -1  # White wins
        else:
            return 0  # Draw
    
    def close(self):
        """Shutdown KataGo"""
        if self.process:
            self._send_command("quit")
            self.process.terminate()
            self.process.wait()
    
    def __del__(self):
        self.close()


def create_katago_config():
    """
    Create minimal KataGo config file
    
    Save this as 'gtp_config.cfg' in your project directory
    """
    config = """
    # Minimal KataGo GTP config for training

    # Log settings
    logFile = katago.log
    logAllGTPCommunication = false
    logSearchInfo = false

    # Performance
    numSearchThreads = 4
    nnCacheSizePowerOfTwo = 20

    # Search settings (will be overridden by kata-set-param)
    maxVisits = 100
    maxPlayouts = 0

    # Rules
    rules = tromp-taylor
    koRule = SIMPLE                                                                                                                                                                                                                                                                     
    scoringRule = AREA
    multiStoneSuicideLegal = false

    # Time control (ignored for genmove commands with visits set)
    maxTime = 10.0
    """
    
    with open("gtp_config.cfg", "w") as f:
        f.write(config)
    
    print("✓ Created gtp_config.cfg")


if __name__ == "__main__":
    # Test KataGo connection
    create_katago_config()
    
    try:
        kata = KataGoOpponent(num_visits=100)
        kata.new_game()
        
        # Test move
        move = kata.get_move("black")
        print(f"KataGo played: {move}")
        
        kata.close()
        print("✓ KataGo test successful")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nSetup instructions:")
        print("1. Download KataGo from: https://github.com/lightvector/KataGo/releases")
        print("2. Ungzip model: gunzip kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz")
        print("3. Place files in project directory")