# sidewalk_ai/models/_obstacles.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------- #
SIDEWALK_LABELS = ("sidewalk", "pavement")  # synonyms
IGNORE_LABELS = {
    "road",
    "route",
    "building",
    "wall",
    "sky",
    "terrain",
    "floor",
    "earth",
    "ceiling",
    "bridge",
    "car",
    "bus",
    "truck",
    "train",
    "motorcycle",
    "bicycle",
    "person",
    "street lamp",
    "fence",
}


def _label_matches(label: str, vocabulary) -> bool:
    """
    Whether *label* names one of the classes in *vocabulary*.

    ADE20K labels are comma-separated synonym lists ("building, edifice"), so
    the first synonym is what gets compared. Testing `startswith` against the
    whole string instead let a class swallow every longer name sharing its
    prefix -- "skyscraper" was being ignored as "sky".
    """
    head = label.split(",", 1)[0].strip()
    return head in vocabulary


MIN_INST_AREA_PX = 30  # reject very tiny noise blobs
MIN_OVERLAP_PX = 10  # at least this many pixels on sidewalk
MIN_OVERLAP_RATIO = 0.01  # ≥ 1 % of the instance must sit on sidewalk
BASE_DILATE_PX = 2  # tolerância p/ contato com calçada (ajuste fino)
BASE_MIN_AREA_PX = 12  # filtro p/ “bases” muito pequenas


def extract_obstacles(
    seg_map: np.ndarray,
    seg_info: Sequence[Tuple[int, str]],
    sidewalk_mask: np.ndarray,
    *,
    ignore_labels: Iterable[str] | None = None,
    sidewalk_labels: Iterable[str] | None = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Return [(label, bool-mask), …] onde a máscara é a **base** do obstáculo
    (interseção com a calçada), instanciada por componentes conexos.

    *ignore_labels* and *sidewalk_labels* default to the module constants, which
    were written against ADE20K's 150 classes. A back-end with a different label
    space passes its own: Cityscapes has 19 coarse classes, so a single shared
    vocabulary does not describe both. See `docs/reproducibility.md`.
    """
    ignore = frozenset(IGNORE_LABELS if ignore_labels is None else ignore_labels)
    sidewalk_vocab = frozenset(SIDEWALK_LABELS if sidewalk_labels is None else sidewalk_labels)
    obstacles: list[Tuple[str, np.ndarray]] = []
    H, W = seg_map.shape
    # dilatar levemente a calçada para tolerar pequenos desalinhamentos
    if BASE_DILATE_PX > 0:
        k = 2 * BASE_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sidewalk_base = cv2.dilate(sidewalk_mask.astype(np.uint8), kernel, iterations=1).astype(
            bool
        )
    else:
        sidewalk_base = sidewalk_mask.astype(bool)

    for seg_id, raw_lbl in seg_info:
        lbl = raw_lbl.lower().strip()

        # 1) skip labels we don't care about
        if _label_matches(lbl, ignore) or _label_matches(lbl, sidewalk_vocab):
            continue

        inst_mask = seg_map == seg_id
        inst_area = int(inst_mask.sum())

        # 2) basic sanity filters
        if inst_area < MIN_INST_AREA_PX:
            # print(f"Skipping {raw_lbl} ({lbl}) due to area {inst_area}")
            continue

        # 3) base = contato com a calçada (com leve dilatação)
        base_mask = inst_mask & sidewalk_base
        base_px = int(base_mask.sum())
        base_r = base_px / max(1, inst_area)
        if base_px < MIN_OVERLAP_PX or base_r < MIN_OVERLAP_RATIO:
            # sem base → não é obstáculo “na calçada”
            continue

        # 4) instanciar pela BASE (componentes conexos na base)
        n, lab = cv2.connectedComponents(base_mask.astype(np.uint8), connectivity=8)
        for cid in range(1, n):
            comp_base = lab == cid
            if int(comp_base.sum()) < BASE_MIN_AREA_PX:
                continue
            # máscara final do obstáculo = a PRÓPRIA BASE (robusta à copa colada)
            # (opcional: dilatar 1 px para visualizar melhor)
            obstacles.append((f"{raw_lbl}#{seg_id}:base{cid}", comp_base.astype(bool)))

    return obstacles
