"""
Parse Go Dataset from SGF files and convert to training format.
Supports multi-part archives - now properly merges before extracting.
"""

import os
import py7zr
import numpy as np
from typing import List, Tuple, Optional
import re
from pathlib import Path
import pickle
import shutil

from board_fast import FastGoBoard
from model import board_to_input


##############################################################
# 1) SGF PARSER
##############################################################

class SGFParser:
    def __init__(self):
        self.board_size = 19

    def coord_to_move(self, sgf_coord):
        if not sgf_coord or len(sgf_coord) != 2:
            return None
        col = ord(sgf_coord[0]) - ord('a')
        row = ord(sgf_coord[1]) - ord('a')
        return (row, col) if 0 <= row < 19 and 0 <= col < 19 else None

    def parse_sgf_file(self, filepath):
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            result = re.search(r'RE\[(.*?)\]', content)
            winner = None
            if result:
                r = result.group(1)
                winner = 1 if "B+" in r else -1 if "W+" in r else None

            black_moves = re.findall(r';B\[(.*?)\]', content)
            white_moves = re.findall(r';W\[(.*?)\]', content)

            if len(black_moves) + len(white_moves) < 50:
                return None

            moves = []
            for i, m in enumerate(black_moves): moves.append((i*2, 1, self.coord_to_move(m)))
            for i, m in enumerate(white_moves): moves.append((i*2+1,-1, self.coord_to_move(m)))
            moves.sort(key=lambda x:x[0])

            return {"moves":[(p,m)for _,p,m in moves],"winner":winner}
        except:
            return None


##############################################################
# 2) CONVERTER
##############################################################

class DatasetConverter:
    def game_to_training_data(self, game):
        board = FastGoBoard(19)
        samples=[]; winner=game["winner"]
        if winner is None: return []

        for player, move in game["moves"]:
            state = board_to_input(board.board, player, board.get_history())
            policy = np.zeros(362,dtype=np.float32)
            policy[361 if move is None else move[0]*19+move[1]]=1.0
            value = 1.0 if winner==player else -1.0

            samples.append((state,policy,value))
            board.play_move(move,player)

        return samples


##############################################################
# 3) MERGE AND EXTRACT MULTI-PART .7z FILES (FIXED)
##############################################################

def merge_multipart_files(base_path, output_path):
    """Merge .7z.001, .7z.002, ... into a single .7z file"""
    print(f"   🔗 Merging parts into {output_path}")
    
    with open(output_path, 'wb') as outfile:
        part_num = 1
        while True:
            part_file = f"{base_path}.{part_num:03d}"
            if not os.path.exists(part_file):
                break
            
            print(f"      Adding {os.path.basename(part_file)}")
            with open(part_file, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
            part_num += 1
    
    return part_num > 1  # Return True if we found parts


def extract_7z_archives(src, out):
    os.makedirs(out, exist_ok=True)
    
    # Find all unique base names for multi-part archives
    multipart_bases = set()
    single_archives = []
    
    for file in os.listdir(src):
        path = os.path.join(src, file)
        
        if file.endswith(".7z.001"):
            # Extract base name (e.g., "1k.7z" from "1k.7z.001")
            base_name = file[:-4]  # Remove ".001"
            multipart_bases.add(base_name)
        elif file.endswith(".7z") and not any(file.startswith(base[:-3]) for base in multipart_bases):
            single_archives.append(path)
    
    # Process multi-part archives
    for base_name in multipart_bases:
        print(f"\n📦 Processing multi-part archive: {base_name}")
        base_path = os.path.join(src, base_name)
        merged_path = os.path.join(src, f"{base_name}_merged.7z")
        
        try:
            # Merge parts
            if merge_multipart_files(base_path, merged_path):
                # Extract merged file
                print(f"   📂 Extracting merged archive...")
                with py7zr.SevenZipFile(merged_path, 'r') as z:
                    z.extractall(out)
                print("   ✔ Extraction complete")
                
                # Clean up merged file
                os.remove(merged_path)
            else:
                print("   ❗ No parts found")
        except Exception as e:
            print(f"   ❗ FAILED: {e}")
    
    # Process single archives
    for path in single_archives:
        file = os.path.basename(path)
        print(f"\n📦 Extracting {file} ...")
        try:
            with py7zr.SevenZipFile(path, 'r') as z:
                z.extractall(out)
            print("   ✔ Extraction complete")
        except Exception as e:
            print(f"   ❗ FAILED: {e}")


##############################################################
# 4) PROCESS SGF → TENSOR FORMAT
##############################################################

def process_dataset(sgf_dir, outfile, max_games=None):
    parser = SGFParser()
    conv = DatasetConverter()
    data = []
    sgf = list(Path(sgf_dir).rglob("*.sgf"))

    print(f"\n📁 Found {len(sgf)} SGF files")
    
    if len(sgf) == 0:
        print("   ❗ No SGF files found - check extraction")
        return []
    
    for i, f in enumerate(sgf):
        if max_games and i >= max_games:
            break
        
        game = parser.parse_sgf_file(str(f))
        if not game:
            continue
        
        samples = conv.game_to_training_data(game)
        data += samples
        
        if (i + 1) % 200 == 0:
            print(f"   → {i+1} games processed, {len(data)} samples")

    print(f"\n🏁 Total samples: {len(data)} from {len(sgf)} games")
    print(f"   Saving → {outfile}")
    
    with open(outfile, "wb") as f:
        pickle.dump(data, f)
    
    print("   ✔ Done")
    return data


##############################################################
# 5) RUN
##############################################################

##############################################################
# 6) AUTO-DETECT PROJECT ROOT
##############################################################

def find_project_root():
    """Find the GOAI project root directory"""
    current = Path.cwd()
    
    # Check if we're IN go-dataset folder (has 1k, 2k subdirs)
    if current.name == "go-dataset" and (current / "1k").exists():
        # Go up one level to actual project root
        return current.parent
    
    # Check if we're already in project root (has go-dataset folder)
    if (current / "go-dataset" / "1k").exists():
        return current
    
    # Check if we're in a nested folder inside go-dataset
    for parent in current.parents:
        if parent.name == "go-dataset" and (parent / "1k").exists():
            return parent.parent
    
    # Search upwards for model.py and go-dataset
    for parent in [current] + list(current.parents):
        if (parent / "model.py").exists() and (parent / "go-dataset" / "1k").exists():
            return parent
    
    return None


if __name__ == "__main__":
    
    print("=" * 60)
    print("FOXQ GO DATASET PROCESSOR")
    print("=" * 60)
    
    # Find project root
    project_root = find_project_root()
    
    if project_root is None:
        print("\n❌ ERROR: Cannot find project root!")
        print("   Please run this script from the GOAI project directory")
        print(f"   Current directory: {Path.cwd()}")
        exit(1)
    
    print(f"\n📂 Project root: {project_root}")
    os.chdir(project_root)
    
    # Set up paths
    archive_dir = project_root / "go-dataset" / "1k"
    extract_dir = project_root / "go-dataset" / "extracted"
    output_file = project_root / "processed_go_dataset.pkl"
    
    # Verify archive directory exists
    if not archive_dir.exists():
        print(f"\n❌ ERROR: Archive directory not found: {archive_dir}")
        print("   Expected structure:")
        print("   GOAI/")
        print("   ├── go-dataset/")
        print("   │   └── 1k/")
        print("   │       ├── 1k.7z.001")
        print("   │       ├── 1k.7z.002")
        print("   │       └── ...")
        exit(1)

    print("\n🔧 STEP 1 — Extracting archives...")
    extract_7z_archives(str(archive_dir), str(extract_dir))

    print("\n🔧 STEP 2 — Converting SGF files...")
    data = process_dataset(str(extract_dir), str(output_file), max_games=1000)

    print("\n🔧 STEP 3 — Verification")
    if data:
        print(f"✅ SUCCESS: {len(data)} training samples ready")
        print(f"   Sample shape: state={data[0][0].shape}, policy={data[0][1].shape}, value={data[0][2]}")
    else:
        print("❌ FAILED: No data extracted")
        print("   Check that archives are valid and contain .sgf files")