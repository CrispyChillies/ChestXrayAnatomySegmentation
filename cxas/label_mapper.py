import json, os
import colorcet as cc
from pathlib import Path

this_directory = Path(__file__).parent
paxray_labels = json.load(
    open(os.path.join(str(this_directory), "data/paxray_labels.json"))
)
id2label_dict = paxray_labels["label_dict"]

thoracic_spine = [
    "thoracic spine",
    "vertebrae T1",
    "vertebrae T2",
    "vertebrae T3",
    "vertebrae T4",
    "vertebrae T5",
    "vertebrae T6",
    "vertebrae T7",
    "vertebrae T8",
    "vertebrae T9",
    "vertebrae T10",
    "vertebrae T11",
    "vertebrae T12",
]

all_vertebrae = [
    "vertebrae L1",
    "vertebrae L2",
    "vertebrae L3",
    "vertebrae L4",
    "vertebrae L5",
    "vertebrae T1",
    "vertebrae T2",
    "vertebrae T3",
    "vertebrae T4",
    "vertebrae T5",
    "vertebrae T6",
    "vertebrae T7",
    "vertebrae T8",
    "vertebrae T9",
    "vertebrae T10",
    "vertebrae T11",
    "vertebrae T12",
    "vertebrae C1",
    "vertebrae C2",
    "vertebrae C3",
    "vertebrae C4",
    "vertebrae C5",
    "vertebrae C6",
]

cervical_spine = [
    "vertebrae C1",
    "vertebrae C2",
    "vertebrae C3",
    "vertebrae C4",
    "vertebrae C5",
    "vertebrae C6",
]

lumbar_spine = [
    "lumbar spine",
    "vertebrae L1",
    "vertebrae L2",
    "vertebrae L3",
    "vertebrae L4",
    "vertebrae L5",
]

clavicle_set = ["clavicles", "clavicle left", "clavicle right"]

scapula_set = [
    "scapulas",
    "scapula left",
    "scapula right",
]

humerus = [
    "humerus",
    "humerus left",
    "humerus right",
]

rib = [
    "posterior 12th rib right",
    "posterior 12th rib left",
    "anterior 11th rib right",
    "posterior 11th rib right",
    "anterior 11th rib left",
    "posterior 11th rib left",
    "anterior 10th rib right",
    "posterior 10th rib right",
    "anterior 10th rib left",
    "posterior 10th rib left",
    "anterior 9th rib right",
    "posterior 9th rib right",
    "anterior 9th rib left",
    "posterior 9th rib left",
    "anterior 8th rib right",
    "posterior 8th rib right",
    "anterior 8th rib left",
    "posterior 8th rib left",
    "anterior 7th rib right",
    "posterior 7th rib right",
    "anterior 7th rib left",
    "posterior 7th rib left",
    "anterior 6th rib right",
    "posterior 6th rib right",
    "anterior 6th rib left",
    "posterior 6th rib left",
    "anterior 5th rib right",
    "posterior 5th rib right",
    "anterior 5th rib left",
    "posterior 5th rib left",
    "anterior 4th rib right",
    "posterior 4th rib right",
    "anterior 4th rib left",
    "posterior 4th rib left",
    "anterior 3rd rib right",
    "posterior 3rd rib right",
    "anterior 3rd rib left",
    "posterior 3rd rib left",
    "anterior 2nd rib right",
    "posterior 2nd rib right",
    "anterior 2nd rib left",
    "posterior 2nd rib left",
    "anterior 1st rib right",
    "posterior 1st rib right",
    "anterior 1st rib left",
    "posterior 1st rib left",
]

ribsuper = [
    "12th rib",
    "12th rib",
    "anterior 11th rib",
    "posterior 11th rib",
    "anterior 10th rib",
    "posterior 10th rib",
    "anterior 9th rib",
    "posterior 9th rib",
    "anterior 8th rib",
    "posterior 8th rib",
    "anterior 7th rib",
    "posterior 7th rib",
    "anterior 6th rib",
    "posterior 6th rib",
    "anterior 5th rib",
    "posterior 5th rib",
    "anterior 4th rib",
    "posterior 4th rib",
    "anterior 3rd rib",
    "posterior 3rd rib",
    "anterior 2nd rib",
    "posterior 2nd rib",
    "anterior 1st rib",
    "posterior 1st rib",
]

diaphragm = [
    "diaphragm",
    "left hemidiaphragm",
    "right hemidiaphragm",
]

mediastinum = [
    "cardiomediastinum",
    "upper mediastinum",
    "lower mediastinum",
    "anterior mediastinum",
    "middle mediastinum",
    "posterior mediastinum",
]

abdomen = [
    "stomach",
    "small bowel",
    "duodenum",
    # "gallbladder",
    "liver",
    "pancreas",
    "kidney left",
    "kidney right",
]

heart = [
    "heart",
    "heart atrium left",
    "heart atrium right",
    "heart myocardium",
    "heart ventricle left",
    "heart ventricle right",
]

breast = [
    # "breast",
    "breast left",
    "breast right",
]

trachea = [
    "trachea",
    "tracheal bifurcation",
]

zones = [
    "right upper zone lung",
    "right mid zone lung",
    "right lung base",
    "right apical zone lung",
    "left upper zone lung",
    "left mid zone lung",
    "left lung base",
    "left apical zone lung",
]

lung_halves = [
    "right lung",
    "left lung",
]
vessels = [
    "heart",
    "ascending aorta",
    "descending aorta",
    "aortic arch",
    "pulmonary artery",
    "inferior vena cava",
]

lobes = [
    "lung lower lobe left",
    "lung upper lobe left",
    "lung lower lobe right",
    "lung middle lobe right",
    "lung upper lobe right",
]


label_mapper = {id2label_dict[k]: [int(k)] for k in id2label_dict.keys()}

label_mapper = {
    **label_mapper,
    **{
        "thoracic spine": [label_mapper[i] for i in thoracic_spine],
        "all vertebrae": [label_mapper[i] for i in all_vertebrae],
        "cervical spine": [label_mapper[i] for i in cervical_spine],
        "lumbar spine": [label_mapper[i] for i in lumbar_spine],
        "clavicle set": [label_mapper[i] for i in clavicle_set],
        "scapula set": [label_mapper[i] for i in scapula_set],
        "ribs": [label_mapper[i] for i in rib],
        "ribs super": [label_mapper[i] for i in ribsuper],
        "diaphragm": [label_mapper[i] for i in diaphragm],
        "mediastinum": [label_mapper[i] for i in mediastinum],
        "abdomen": [label_mapper[i] for i in abdomen],
        "heart region": [label_mapper[i] for i in heart],
        "breast tissue": [label_mapper[i] for i in breast],
        "trachea": [label_mapper[i] for i in trachea],
        "lung zones": [label_mapper[i] for i in zones],
        "lung halves": [label_mapper[i] for i in lung_halves],
        "vessels": [label_mapper[i] for i in vessels],
        "lung lobes": [label_mapper[i] for i in lobes],
    },
}

def _normalize_color(color_value):
    if isinstance(color_value, str):
        color_value = color_value.lstrip("#")
        if len(color_value) == 6:
            color_value += "FF"
        if len(color_value) != 8:
            raise ValueError(f"Unsupported color format: {color_value}")
        return tuple(int(color_value[i : i + 2], 16) / 255.0 for i in range(0, 8, 2))

    if isinstance(color_value, (list, tuple)):
        if len(color_value) == 3:
            color_value = list(color_value) + [1.0]
        elif len(color_value) != 4:
            raise ValueError(f"Unsupported color format: {color_value}")

        if any(component > 1 for component in color_value):
            return tuple(float(component) / 255.0 for component in color_value)
        return tuple(float(component) for component in color_value)

    raise ValueError(f"Unsupported color format: {color_value}")


def _get_label_colors(num_colors: int):
    cmap = getattr(getattr(cc, "cm", None), "glasbey_bw_minc_20", None)
    if callable(cmap):
        return [cmap(i) for i in range(num_colors)]

    palette = getattr(cc, "glasbey_bw_minc_20", None)
    if palette:
        return [_normalize_color(palette[i % len(palette)]) for i in range(num_colors)]

    raise RuntimeError(
        "colorcet does not expose 'glasbey_bw_minc_20' as a callable colormap or palette."
    )


colors = _get_label_colors(len(id2label_dict.keys()))
colors_alpha = [[i[0], i[1], i[2], i[3] / 2] for i in colors]
category_colors = {colors[i][:3]: i for i in range(len(colors))}
category_ids = {id2label_dict[k]: int(k) for k in id2label_dict.keys()}
