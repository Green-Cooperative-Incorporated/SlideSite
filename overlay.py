import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from WSI_Preprocessing.Preprocessing.WSI_Scanning import readWSI
from WSI_Preprocessing.Preprocessing.Denoising import denoising
from WSI_Preprocessing.Preprocessing.Patch_extraction_creatia import patch_extraction_random, all_patches_extarction
from WSI_Preprocessing.Preprocessing.Utilities import stainremover_small_patch_remover1
import openslide
from openslide import (OpenSlide, OpenSlideError,OpenSlideUnsupportedFormatError)
import os
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
torch.cuda.empty_cache()
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected
from sklearn.metrics import silhouette_samples

#tile_path = "/data/home/sai/Desktop/Vision_Transformer/data/tiles_pickles2/TCGA-DB-A64R-01Z-00-DX1/TCGA-DB-A64R-01Z-00-DX1_tile_0002.pkl"
tiles_path = "/data/home/sai/Desktop/Vision_Transformer/data/luad_tiles/TCGA-05-4244-01Z-00-DX1"
patch_size = 256
n_clusters = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load in tile
# with open(tiles_path + tile, 'rb') as f:
#     data = pickle.load(f)
#tile_img = data['whole_tile'] # readWSI(svs_path)
path = "/data/home/sai/Desktop/Vision_Transformer/data/camelyon-dataset/CAMELYON16/images/tumor_009.tif"
# tile_img, slidedim = readWSI(path, "20x", None, None, None)
# if not isinstance(tile_img, torch.Tensor):
#     tile_img = torch.from_numpy(tile_img)
# tile_tensor = tile_img.permute(2, 0, 1).float().unsqueeze(0).to(device)  # (1, 3, H, W)

def extractingPatches(
    inputsvs,
    tile_size=2048,
    patch_size=(256, 256),
    Annotation=None,
    Annotatedlevel=0,
    Requiredlevel=0
):
    """
    1) Load the whole‐slide image (`slide1`) at 20× via readWSI(...)
    2) Tile the slide in blocks of up to tile_size × tile_size, covering partial
       tiles at the right/bottom edges as well.
    3) For each tile, scan all possible non‐overlapping patch_size×patch_size positions
       (row‐by‐row, column‐by‐column). If stainremover_small_patch_remover1(...) returns
       a non‐None array, mark that as a “valid patch.”
    4) If a given tile yields exactly 10 valid patches, we call that tile “valid,”
       store the 10 patch arrays + their bounding boxes, and move on.
    5) Return a dict-of‐dicts: { slideID : { tile_num : [ { 'patch_array':…, 'bbox':(r0,c0,r1,c1) },… ] } }.

    - slideID is simply the first 16 characters of the input file path.
    - tile_num increments in strictly row‐major order, starting at 0.
    """
    output_tiles_dir = "tiles_pickles2"
    os.makedirs(output_tiles_dir, exist_ok=True)

    tileDict = []
    slide1, slidedim = readWSI(inputsvs, "20x", Annotation, Annotatedlevel, Requiredlevel)
    slideID = os.path.basename(inputsvs).split('.')[0]

    output_dir = os.path.join(output_tiles_dir, slideID)
    os.makedirs(output_dir, exist_ok=True)

    print("######################## loaded models and slide #################################")
    print("Shape of the slide:", slide1.shape)

    tile_num = 0
    valid_tile_num = 0
    H, W = slide1.shape[:2]

    # Loop over every tile in row‐major order, covering edges
    for row_start in range(0, H, tile_size):
        for col_start in range(0, W, tile_size):
            # Determine how far this tile actually extends (clamp at edges)
            row_end_tile = min(row_start + tile_size, H)
            col_end_tile = min(col_start + tile_size, W)

            tile_height = row_end_tile - row_start
            tile_width  = col_end_tile - col_start
            max_k = tile_height // patch_size[0]
            all_patches_in_this_tile = []
            patch_num = 0

            # Within this tile, try to extract up to 10 valid patches
            while patch_num < 32:
                found_one_this_iter = False

                # Build a list of all (k, l) positions in this tile exactly once
                max_k = tile_height // patch_size[0]
                max_l = tile_width  // patch_size[1]
                if max_k == 0 or max_l == 0:
                    tile_num += 1
                    continue

                all_positions = [(k, l) for k in range(max_k) for l in range(max_l)]
                # Optionally shuffle if you want random sampling order:
                # import random
                # random.shuffle(all_positions)

                for k, l in all_positions:
                    if patch_num >= 32:
                        
                        break

                    center_row = int(row_start + (k * patch_size[0] + patch_size[0]/2))
                    center_col = int(col_start + (l * patch_size[1] + patch_size[1]/2))
                    half_h = patch_size[0] // 2
                    half_w = patch_size[1] // 2
                    r0, r1 = center_row - half_h, center_row + half_h
                    c0, c1 = center_col - half_w, center_col + half_w

                    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                        continue

                    sample_img = slide1[r0:r1, c0:c1]
                    patchs = stainremover_small_patch_remover1(sample_img, patch_size)
                    if patchs is not None:
                        all_patches_in_this_tile.append({
                            'patch_array': patchs,
                            'bbox': (r0, c0, r1, c1)
                        })
                        print(f"Patch #{patch_num} found in tile {tile_num} at bbox=({r0},{c0},{r1},{c1})")
                        patch_num += 1

                    if found_one_this_iter:
                        break

                if not found_one_this_iter:
                    # No further valid patches in this tile
                    print(f"No more valid patches found in tile {tile_num}; needed 10, found {patch_num}.")
                    break

            # Only keep this tile if exactly 10 valid patches were found
        
            valid_tile_num += 1
            print(f"Tile {tile_num} is valid (found {patch_num} patches). Total valid tiles so far: {valid_tile_num}")
            # add the tile to list here
            #tileDict[tile_num] = all_patches_in_this_tile

            # save entire tile as a numpy array
            tile_array = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile_array = slide1[row_start:row_end_tile, col_start:col_end_tile]
            tileDict.append(tile_array)

            # save the tile data as a pickle
            tile_data = {
                'tile_num': tile_num,
                'whole_tile': tile_array,
            }

            filename = f"{slideID}_tile_{tile_num:04d}.pkl"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(tile_data, f)
                print(f"Saved tile {tile_num} data to {filepath}")

            tile_num += 1

    print(f"Finished extracting. Total tiles processed: {tile_num}, valid tiles: {valid_tile_num}")
    return tileDict
tile_list = extractingPatches(path,
    tile_size=2048,
    patch_size=(256, 256),
    Annotation=None,
    Annotatedlevel=0,
    Requiredlevel=0)
with open('LUAD_overlay_dict.pkl', 'wb') as f:
    pickle.dump(tile_list, f)
    

# ====== Parameters ======
DATA_DIR = "/data/home/sai/Desktop/Vision_Transformer/data/luad_tiles_sampled"
LABEL_FILE = "/data/home/sai/Desktop/Vision_Transformer/TCGA_LUAD_pancancer atlas 2018_514+8 case_240726_rec risk.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLUSTERS = 6
IMG_SIZE = 2048
PATCH_SIZE = 256
EMBED_DIM = 128
BATCH_SIZE = 5
KNN = N_CLUSTERS
LR = 0.0001
EPOCHS = 200
KLD_WEIGHT = 0.1
EPS           = 1e-8  # small constant to avoid zero probabilities
bins = 10
# ====== ViT Backbone ======
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth=4, heads=4, mlp_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)

    def forward(self, x):
        B = x.size(0)
        patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed
        _ = self.transformer(x)
        return patches

# ====== Graph Encoder ======
class TileGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=1)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        slide_repr = global_mean_pool(x, data.batch)
        logits = self.fc(slide_repr)
        return x, logits
def build_knn_graph(tile_feats, knn=KNN):
    """
    Builds a KNN edge_index for PyG.  
    If num_tiles < 2, returns an empty edge_index.
    Otherwise uses at most (num_tiles - 1) neighbors per node.
    """
    num_tiles = tile_feats.size(0)
    # need at least 2 tiles to build edges
    if num_tiles < 2:
        # empty [2, 0] tensor of long indices
        return torch.empty((2, 0), dtype=torch.long, device=tile_feats.device)

    # don't request more neighbors than (num_tiles - 1)
    n_neighbors = min(knn, num_tiles - 1)

    # compute on CPU
    feats_cpu = tile_feats.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(feats_cpu)
    idxs = nbrs.kneighbors(return_distance=False)

    # build edge list
    edges = []
    for i, nbr in enumerate(idxs):
        for j in nbr:
            # skip self-loop if you want, or include it
            if i != j:
                edges.append([i, j])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=tile_feats.device)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index.to(tile_feats.device)
from sklearn.metrics import silhouette_samples
import numpy as np
import torch
import matplotlib.cm as cm

def overlay_by_silhouette(tile_tensor, node_emb_np, labels, patch_size=256, alpha=0.5, eps=1e-8):
    """
    tile_tensor: torch.Tensor or np.ndarray of shape (3, H, W) or (H, W, 3)
    node_emb_np: np.ndarray of shape (N_patches, D)
    labels:      array-like of length N_patches
    patch_size:  size of each square patch
    alpha:       overlay opacity (0–1)
    eps:         small constant to avoid div/0
    """
    # --- 1) Compute silhouette scores and normalize to [0,1] ---
    sil = silhouette_samples(node_emb_np, labels, metric='euclidean')
    sil_min, sil_max = sil.min(), sil.max()
    sil_norm = (sil - sil_min) / (sil_max - sil_min + eps)   # length N_patches

    # --- 2) Prepare the base image as H×W×3 uint8 ---
    if isinstance(tile_tensor, torch.Tensor):
        arr = tile_tensor.detach().cpu().numpy()
    else:
        arr = tile_tensor.copy()
    # transpose if needed
    if arr.ndim == 3 and arr.shape[0] == 3:
        img = np.transpose(arr, (1,2,0))
    else:
        img = arr
    img = img.astype(np.uint8)
    H, W, _ = img.shape

    # --- 3) Overlay heatmap per patch ---
    overlay = img.copy()
    grid = H // patch_size
    cmap = cm.get_cmap('coolwarm')   # blue→red

    for idx, score in enumerate(sil_norm):
        row, col = divmod(idx, grid)
        y0, x0 = row * patch_size, col * patch_size
        patch = overlay[y0:y0+patch_size, x0:x0+patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue
        # RGB in [0,255]
        color = (np.array(cmap(score)[:3]) * 255).astype(np.uint8)
        overlay[y0:y0+patch_size, x0:x0+patch_size] = (
            (1 - alpha) * patch + alpha * color
        ).astype(np.uint8)

    return overlay

with open('LUAD_overlay_dict.pkl', 'rb') as f:
    tile_list = pickle.load(f)
from sklearn.neighbors import NearestNeighbors
# try GPU, else CPU
try:
    vit = ViT(IMG_SIZE, PATCH_SIZE, EMBED_DIM).to(DEVICE)
    graph_enc = TileGraphEncoder(EMBED_DIM, hidden_dim=64).to(DEVICE)
except RuntimeError as e:
    if 'out of memory' in str(e):
        print("GPU OOM; switching to CPU.")
        torch.cuda.empty_cache()
        DEVICE = torch.device('cpu')
        vit   = ViT(IMG_SIZE, PATCH_SIZE, EMBED_DIM).to(DEVICE)
        graph_enc = TileGraphEncoder(EMBED_DIM, hidden_dim=64).to(DEVICE)
    else:
        raise

checkpoint = torch.load("best_checkpoint.pth", map_location=DEVICE)
# epoch      = checkpoint['epoch'] + 1   # resume at next epoch
vit.load_state_dict(checkpoint['vit_state_dict'])
graph_enc.load_state_dict(checkpoint['graph_state_dict'])
processed_tiles = []
vit.eval()
graph_enc.eval()
expected_shape = (IMG_SIZE, IMG_SIZE, 3)  # adjust if different
expected_shape = (2048, 2048, 3)
patch_size     = 256
n_clusters     = 6
tile_rows      = 22                   # adjust to your grid
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for tile in tile_list:
    if isinstance(tile, np.ndarray):
        arr = tile
        tensor = torch.from_numpy(tile)
    else:
        tensor = tile
        arr = tile.cpu().numpy() if isinstance(tile, torch.Tensor) else np.array(tile)

    if tuple(arr.shape) == expected_shape:
        tile_tensor = tensor.permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. Extract patch embeddings from ViT
            patches = vit(tile_tensor).squeeze(0)  # shape [N_patches, D]

            # 2. Build KNN graph from patch features
            tile_feats = patches.to(device)
            edge_index = build_knn_graph(tile_feats, knn=KNN)
            edge_index = to_undirected(edge_index)

            data = Data(x=tile_feats, edge_index=edge_index)
            node_emb, _ = graph_enc(data)  # node_emb: [N_patches, D]

            # 3. Cluster graph node embeddings
            node_emb_np = node_emb.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_emb_np)
            labels = kmeans.labels_  # [N_patches]

        # 4. Overlay
        overlaid = overlay_by_silhouette(
            tensor.squeeze(0),   # (3,H,W) tile tensor
            node_emb_np,
            labels,
            patch_size=PATCH_SIZE,
            alpha=0.5
        )
        processed_tiles.append(overlaid)

    else:
        # Small tile fallback
        if isinstance(arr, torch.Tensor):
            arr = arr.permute(1, 2, 0).cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        processed_tiles.append(arr.astype(np.uint8))

# === Stitching with variable tile sizes ===

tile_rows = 22  # or whatever your layout is
tile_cols = int(np.ceil(len(processed_tiles) / tile_rows))

# 1) group into rows
rows = [
    processed_tiles[r * tile_cols:(r + 1) * tile_cols]
    for r in range(tile_rows)
]

# 2) compute max heights per row, max widths per column
row_heights = [max(tile.shape[0] for tile in row) for row in rows]
col_widths = [max(rows[r][c].shape[1] for r in range(tile_rows) if c < len(rows[r]))
              for c in range(tile_cols)]

# 3) prepare canvas
full_H = sum(row_heights)
full_W = sum(col_widths)
full_image = np.zeros((full_H, full_W, 3), dtype=np.uint8)

# 4) paste tiles
y_off = 0
for r, row_tiles in enumerate(rows):
    x_off = 0
    for c, tile in enumerate(row_tiles):
        h, w, _ = tile.shape
        full_image[y_off:y_off + h, x_off:x_off + w] = tile
        x_off += col_widths[c]
    y_off += row_heights[r]

# 5) display
plt.figure(figsize=(12, 12))
plt.imshow(full_image)
plt.axis('off')
plt.title("Reconstructed WSI with Graph Encoder Clusters")
plt.tight_layout()
plt.save('thumbnail.png')
