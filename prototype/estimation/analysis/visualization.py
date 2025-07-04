import numpy as np
import os, cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def plot_3d_point_cloud(points_3d, title="3D Point Cloud"):
    """
    points_3d: a NumPy array of shape (N, 3) containing [X, Y, Z] coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2],
        s=2  # marker size
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.show()

    # If you want to save as an image file (e.g. PNG), you can do:
    # plt.savefig("my_point_cloud.png")

def save_debug_masks(debug_dir, base_filename, img_rgb, 
                    ensemble_model1, ensemble_model2,
                    ensemble_mask1, ensemble_mask2,
                    ensemble_mask1_only, ensemble_mask2_only, 
                    fused_mask, agree):
    """
    Salva máscaras individuais para debug.
    """
    
    base_name = os.path.splitext(base_filename)[0]
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Imagem original
    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Model1 mask
    axes[0,1].imshow(ensemble_mask1, cmap='gray')
    axes[0,1].set_title(f'{ensemble_model1}\n({np.sum(ensemble_mask1)} pixels)')
    axes[0,1].axis('off')
    
    # Model2 mask
    axes[0,2].imshow(ensemble_mask2, cmap='gray')
    axes[0,2].set_title(f'{ensemble_model2}\n({np.sum(ensemble_mask2)} pixels)')
    axes[0,2].axis('off')
    
    # Fused mask
    axes[0,3].imshow(fused_mask, cmap='gray')
    axes[0,3].set_title(f'Fused (OR)\n({np.sum(fused_mask)} pixels)')
    axes[0,3].axis('off')
    
    # Model1 only
    axes[1,0].imshow(ensemble_mask2_only, cmap='Reds')
    axes[1,0].set_title(f'{ensemble_model2} Only\n({np.sum(ensemble_mask2_only)} pixels)')
    axes[1,0].axis('off')
    
    # Model2 only
    axes[1,1].imshow(ensemble_mask1_only, cmap='Blues')
    axes[1,1].set_title(f'{ensemble_model1} Only\n({np.sum(ensemble_mask1_only)} pixels)')
    axes[1,1].axis('off')
    
    # Agreement
    axes[1,2].imshow(agree, cmap='Greens')
    axes[1,2].set_title(f'Both Agree\n({np.sum(agree)} pixels)')
    axes[1,2].axis('off')
    
    # Stats text
    total_of = np.sum(ensemble_mask1)
    total_dl = np.sum(ensemble_mask2)
    total_agree = np.sum(agree)
    
    if total_of > 0 and total_dl > 0:
        overlap_ratio = total_agree / min(total_of, total_dl) * 100
        jaccard = total_agree / (total_of + total_dl - total_agree) * 100
    else:
        overlap_ratio = 0
        jaccard = 0
    
    stats_text = f"""Statistics:
    {ensemble_model1}: {total_of:,} pixels
    {ensemble_model2}: {total_dl:,} pixels
    Overlap: {total_agree:,} pixels
    Overlap ratio: {overlap_ratio:.1f}%
    Jaccard IoU: {jaccard:.1f}%"""
    
    axes[1,3].text(0.1, 0.5, stats_text, fontsize=10, 
                   verticalalignment='center', transform=axes[1,3].transAxes)
    axes[1,3].axis('off')
    
    plt.tight_layout()
    debug_path = os.path.join(debug_dir, f"{base_name}_debug.png")
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug masks saved to: {debug_path}")

def show_or_save_sidewalk_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    save_path: str | Path | None = None,
    window_name: str = "Sidewalk Mask Debug"
) -> None:
    
    """
    Overlay a binary sidewalk mask on the original image for debugging.

    Parameters
    ----------
    img_rgb : np.ndarray
        Original RGB image (H×W×3, uint8).
    mask : np.ndarray
        Binary mask (H×W, values 0/1 or 0/255).
    alpha : float
        Transparency of the overlay colour.
    save_path : str | Path | None
        If given, writes PNG to this path instead of showing a window.
    window_name : str
        Title of the OpenCV window when save_path is None.
    """
    # ── normalise mask to {0,255} ────────────────────────────────────────────────
    m = (mask > 0).astype(np.uint8) * 255

    # ── build a coloured overlay (cyan) ─────────────────────────────────────────
    overlay = np.zeros_like(img_rgb)
    overlay[..., 1] = m                # G
    overlay[..., 2] = m                # R   (R+G ⇒ yellow/cyan depending on alpha)

    # ── blend with original ─────────────────────────────────────────────────────
    blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)

    # ── save or display ─────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.is_dir():                  # auto-generate filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_path / f"debug_mask_{ts}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    else:
        cv2.imshow(window_name, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def draw_horizontal_legend(ax, legend_data):
    """
    legend_data: list of (class_name, [R,G,B]) for each segment
    ax: a matplotlib Axes on which to draw the legend in a horizontal row.
    """
    import matplotlib.patches as mpatches
    import numpy as np

    # remove duplicates if multiple segments share the same class
    unique_legend = {}
    for cls, color in legend_data:
        if cls not in unique_legend:
            unique_legend[cls] = color

    # We place squares horizontally, each 0.2 wide, 0.2 tall, with some margin
    x_offset = 0.0
    for cls, color in unique_legend.items():
        # color is [R,G,B] in 0-255
        color_rgb = np.array(color)/255.0 if np.max(color)>1 else color

        # Draw a small rectangle patch
        rect = mpatches.Rectangle((x_offset, 0.0), 0.2, 0.2,
                                  edgecolor='none', facecolor=color_rgb)
        ax.add_patch(rect)
        # Put text to the right of the square
        ax.text(x_offset+0.25, 0.1, cls, va='center', fontsize=9)

        x_offset += 1.0  # move horizontally for the next patch

    # set x-limits to fit all classes
    ax.set_xlim(0, max(2, x_offset))
    ax.set_ylim(0, 0.3)  # just enough vertical space for squares + text
    ax.axis('off')
    ax.set_title("Segment Legend")

def oneformer_visualize(output_path, img_rgb, seg_map, segments_info, model_name):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from transformers import OneFormerForUniversalSegmentation

    oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(model_name)

    # Build a palette
    palette = []
    np.random.seed(42)
    for i in range(150):
        palette.append(np.random.randint(0, 256, size=3))

    # Create color overlay
    height, width = seg_map.shape
    color_seg = np.zeros((height, width, 3), dtype=np.uint8)

    legend_data = []
    for seg_dict in segments_info:
        seg_id = seg_dict["id"]
        label_id = seg_dict["label_id"]
        class_name = oneformer_model.config.id2label[label_id]
        color = palette[label_id % len(palette)]

        mask = (seg_map == seg_id)
        color_seg[mask] = color
        legend_data.append((class_name, color))

    final_overlay = cv2.addWeighted(img_rgb, 0.5, color_seg, 0.5, 0)

    # Create figure with 2 rows and 1 column:
    #   Row 1: the overlay
    #   Row 2: the legend (shorter)
    fig, (ax_img, ax_legend) = plt.subplots(
        nrows=2, 
        figsize=(10, 8), 
        gridspec_kw={"height_ratios": [5,1]}  # bigger for top, smaller for bottom
    )

    # Top row: show the overlay
    ax_img.imshow(final_overlay)
    ax_img.axis('off')
    ax_img.set_title("OneFormer Panoptic Overlay")

    # Bottom row: draw horizontal legend
    draw_horizontal_legend(ax_legend, legend_data)

    plt.tight_layout()
    plt.savefig(output_path)

def detectron2_visualize(output_path, img_rgb, panoptic_seg, segments_info, cfg):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from detectron2.data import MetadataCatalog

    # Build an overlay + legend_data similarly
    color_seg = np.zeros_like(img_rgb, dtype=np.uint8)
    legend_data = []

    for seg in segments_info:
        seg_id = seg["id"]
        cat_id = seg["category_id"]
        isthing = seg.get("isthing", False)

        if isthing:
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[cat_id]
        else:
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[cat_id]

        color = np.random.randint(0,256,size=3,dtype=np.uint8)
        mask = (panoptic_seg == seg_id)
        color_seg[mask] = color

        legend_data.append((class_name, color))

    final_overlay = cv2.addWeighted(img_rgb,0.5,color_seg,0.5,0)

    fig, (ax_img, ax_legend) = plt.subplots(
        nrows=2,
        figsize=(10,8),
        gridspec_kw={"height_ratios": [5,1]}
    )

    ax_img.imshow(final_overlay)
    ax_img.axis('off')
    ax_img.set_title("Detectron2 Panoptic Overlay")

    draw_horizontal_legend(ax_legend, legend_data)

    plt.tight_layout()
    plt.savefig(output_path)

def ensemble_visualize(out_path, overlay, legend_data):
    """
    Versão melhorada da visualização do ensemble.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, (ax_img, ax_legend) = plt.subplots(
        nrows=2, figsize=(10, 8), gridspec_kw={"height_ratios": [4, 1]}
    )
    
    # Mostrar imagem com overlay
    ax_img.imshow(overlay)
    ax_img.axis('off')
    ax_img.set_title('Ensemble Segmentation Results', fontsize=14, fontweight='bold')
    
    # Criar legenda melhorada
    ax_legend.set_xlim(0, 10)
    ax_legend.set_ylim(0, 1)
    
    x_positions = [1, 4, 7]
    
    for i, (label, color) in enumerate(legend_data):
        # Criar retângulo colorido
        rect = patches.Rectangle((x_positions[i], 0.3), 0.8, 0.4, 
                               facecolor=np.array(color)/255.0, 
                               edgecolor='black', linewidth=1)
        ax_legend.add_patch(rect)
        
        # Adicionar texto
        ax_legend.text(x_positions[i] + 0.4, 0.1, label, 
                      ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax_legend.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Ensemble visualization saved to: {out_path}")
    
def midas_visualize(img_path=None, img_rgb=None, panoptic_seg=None, segments_info=None,
                    backend=None, oneformer_model_name=None, cfg=None,
                    sidewalk_mask=None,
                    ensemble_model1=None,
                    ensemble_model2=None,          
                    ensemble_mask1=None,         
                    ensemble_mask2=None):          
    
    # set main output file
    output_file = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    
    backend = backend.lower()

    if backend == "detectron2":
        #----------------------------------
        # DETECTRON2 VISUALIZATION
        #----------------------------------
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog

        if cfg is None:
            raise ValueError("cfg is required for detectron2 visualization")

        # 'panoptic_seg' is typically a torch.Tensor of shape [H, W]
        # or we might do panoptic_seg.cpu() above.
        # Ensure we have a CPU tensor if needed:
        if hasattr(panoptic_seg, "cpu"):
            panoptic_seg = panoptic_seg.cpu().numpy()

        # specialize path for folder
        output_path = os.path.join("estimation\output\detectron2", output_file)
        detectron2_visualize(output_path, img_rgb, panoptic_seg, segments_info, cfg)

    elif backend == "oneformer":
        #----------------------------------
        # ONEFORMER VISUALIZATION
        #----------------------------------
        if oneformer_model_name is None:
            raise ValueError(f"{oneformer_model_name} is required for OneFormer visualization")

        # If 'panoptic_seg' is a dict from post_process (like {"segmentation": array, "segments_info":...}),
        # we might need to unify it. Check if it's a dict or an array.
        if isinstance(panoptic_seg, dict) and "segmentation" in panoptic_seg:
            seg_map = panoptic_seg["segmentation"]
        else:
            seg_map = panoptic_seg  # assume it's already the raw segmentation array

        # Convert to numpy if needed
        if hasattr(seg_map, "cpu"):
            seg_map = seg_map.cpu().numpy()
        seg_map = np.array(seg_map, dtype=np.int32)

        # specialize path for folder
        output_path = os.path.join("estimation\output\oneformer", output_file)
        oneformer_visualize(output_path, img_rgb, seg_map, segments_info, oneformer_model_name)
    
    elif backend.startswith("ensemble"):
        # ----------------------------------
        # ENSEMBLE VISUALIZATION
        # ----------------------------------
        # seg_map here is the OneFormer map (produced earlier so that we
        # still have segments_info); sidewalk_mask_en is the fused mask.
        # We build a simple overlay that shows         ­
        #   • cyan  pixels = Mask1 only
        #   • magenta pixels = Mask2 only
        #   • yellow pixels  = both agree
        #   • image colours  = background
        #

        # Verificar se as máscaras foram passadas
        if ensemble_mask1 is None or ensemble_mask2 is None:
            raise ValueError("Both masks are mandatory for ensemble view")
        
        # Garantir que todas as máscaras sejam numpy arrays uint8
        fused_mask = sidewalk_mask.astype(np.uint8)
        ensemble_mask1 = ensemble_mask1.astype(np.uint8) 
        ensemble_mask2 = ensemble_mask2.astype(np.uint8)

        # Garantir que todas tenham o mesmo tamanho
        if not (fused_mask.shape == ensemble_mask1.shape == ensemble_mask2.shape):
            print(f"Warning: Mask shapes differ - Fused: {fused_mask.shape}, {ensemble_model1}: {ensemble_mask1.shape}, {ensemble_model2}: {ensemble_mask2.shape}")
            # Redimensionar para o tamanho da fused_mask
            from PIL import Image
            if ensemble_mask1.shape != fused_mask.shape:
                ensemble_mask1_pil = Image.fromarray(ensemble_mask1)
                ensemble_mask1_pil = ensemble_mask1_pil.resize((fused_mask.shape[1], fused_mask.shape[0]), Image.NEAREST)
                ensemble_mask1 = np.array(ensemble_mask1_pil).astype(np.uint8)
            
            if ensemble_mask2.shape != fused_mask.shape:
                ensemble_mask2_pil = Image.fromarray(ensemble_mask2)
                ensemble_mask2_pil = ensemble_mask2_pil.resize((fused_mask.shape[1], fused_mask.shape[0]), Image.NEAREST)
                ensemble_mask2 = np.array(ensemble_mask2_pil).astype(np.uint8)

        # Converter para máscaras booleanas para operações lógicas
        fused_bool = fused_mask.astype(bool)
        ensemble_bool1 = ensemble_mask1.astype(bool)
        ensemble_bool2 = ensemble_mask2.astype(bool)
        
        # Calcular as diferentes regiões
        ensemble_mask1_only = ensemble_bool1 & ~ensemble_bool2  # Mask1 detecta, Mask2 não
        ensemble_mask2_only = ensemble_bool2 & ~ensemble_bool1  # Mask2 detecta, Mask1 não
        agree_mask = ensemble_bool1 & ensemble_bool2  # Ambos detectam (interseção)

        # Debug: mostrar estatísticas
        print(f"Ensemble Visualization Stats:")
        print(f"  {ensemble_model1} only: {np.sum(ensemble_mask1_only)} pixels")
        print(f"  {ensemble_model2} only: {np.sum(ensemble_mask2_only)} pixels") 
        print(f"  Both agree: {np.sum(agree_mask)} pixels")
        print(f"  Total fused: {np.sum(fused_bool)} pixels")

        # Verificar se há concordância
        if np.sum(agree_mask) == 0:
            print("Warning: Nenhuma região de concordância encontrada!")
        
        # Criar overlay colorido
        overlay = img_rgb.copy().astype(np.uint8)
        
        # Aplicar cores com transparência para melhor visualização
        alpha = 0.6  # Transparência
        
        # Mask1 only - Ciano
        overlay[ensemble_mask2_only] = (
            overlay[ensemble_mask2_only] * (1 - alpha) + 
            np.array([0, 255, 255]) * alpha
        ).astype(np.uint8)
        
        # Mask2 only - Magenta  
        overlay[ensemble_mask1_only] = (
            overlay[ensemble_mask1_only] * (1 - alpha) + 
            np.array([255, 0, 255]) * alpha
        ).astype(np.uint8)
        
        # Ambos concordam - Amarelo (aplicado por último para ter prioridade)
        overlay[agree_mask] = (
            overlay[agree_mask] * (1 - alpha) + 
            np.array([255, 255, 0]) * alpha
        ).astype(np.uint8)
        
        # Dados da legenda
        legend_data = [
            ("Mask1 only", (0, 255, 255)),
            ("Mask2 only", (255, 0, 255)),
            ("Both models", (255, 255, 0))
        ]
        
        # Criar pasta se não existir
        output_dir = "estimation/output/ensemble"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        
        # Chamar função de visualização
        ensemble_visualize(output_path, overlay, legend_data)
        
        
        # Salvar máscaras individuais para debug
        debug_output_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_output_dir, exist_ok=True)
        
        save_debug_masks(debug_output_dir, output_file, img_rgb,
                         ensemble_model1, ensemble_model2,
                         ensemble_mask1, ensemble_mask2, 
                         ensemble_mask1_only, ensemble_mask2_only, 
                         fused_mask, agree_mask)

    else:
        raise ValueError(f"Unsupported backend: {backend}")