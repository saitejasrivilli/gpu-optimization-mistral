# Benchmarking Dataset - Conversion Instructions

## DOWNLOAD

**File to Download:** `Benchmarking Dataset.zip` (1.6 GB)

**Direct Link:** https://zenodo.org/records/14827784/files/Benchmarking%20Dataset.zip

---

## DATASET STRUCTURE (After Extraction)

```
Benchmarking Dataset/
├── train/
│   ├── images/           (5,012 images)
│   ├── labelme/          (1,782 JSON files)
│   ├── pascal/           (1,782 JSON files)
│   ├── yolo/             (1,798 label files)
│   └── coco/
│       └── train.json
├── valid/
│   ├── images/           (628 images)
│   ├── pascal/           (215 JSON files)
│   ├── yolo/             (215 label files)
│   └── coco/
│       └── valid.json
└── test/
    ├── images/           (629 images)
    ├── pascal/           (233 JSON files)
    ├── yolo/             (233 label files)
    └── coco/
        └── test.json
```

---

## CONVERSION SCRIPT

Save this as `convert_benchmarking.py` in the root directory of the extracted dataset.

```python
import os
import json
import shutil
from pathlib import Path

def inspect_coco_classes(coco_file):
    """Inspect COCO JSON to understand class structure"""
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📋 Classes found in {Path(coco_file).name}:")
    print("-" * 60)
    for cat in data.get('categories', []):
        print(f"  ID {cat['id']}: {cat['name']}")
    
    return {cat['id']: cat['name'] for cat in data.get('categories', [])}

def map_class_to_11class(class_name):
    """
    Map original class names to 11-class taxonomy.
    Returns (new_class_id, new_class_name) or None if not mappable
    """
    class_name_lower = class_name.lower()
    
    # Class 1: Caries – Non-Cavitated
    if any(x in class_name_lower for x in ['non-cavitated', 'white spot', 'demineralization', 
                                             'early caries', 'initial caries']):
        return (0, 'caries_non_cavitated')
    
    # Class 2: Caries – Cavitated
    if any(x in class_name_lower for x in ['cavitated', 'cavity', 'caries', 'decay', 'dental caries']):
        # Exclude if it's gross/severe
        if not any(x in class_name_lower for x in ['gross', 'severe', 'rotten', 'extensive']):
            return (1, 'caries_cavitated')
    
    # Class 3: Crack / Fracture
    if any(x in class_name_lower for x in ['crack', 'fracture', 'fractured', 'craze']):
        return (2, 'crack_fracture')
    
    # Class 4: Gross Carious Destruction
    if any(x in class_name_lower for x in ['gross', 'severe decay', 'rotten', 'extensive caries',
                                             'badly decayed', 'breakdown']):
        return (3, 'gross_carious_destruction')
    
    # Class 5: Gingivitis
    if any(x in class_name_lower for x in ['gingivitis', 'gum disease', 'gingival inflammation']):
        return (4, 'gingivitis')
    
    # Class 6: Abscess / Fistula
    if any(x in class_name_lower for x in ['abscess', 'fistula', 'periapical abscess', 
                                             'apical abscess']):
        return (5, 'abscess_fistula')
    
    # Class 7: Ulcer / Lesion
    if any(x in class_name_lower for x in ['ulcer', 'lesion', 'aphthous', 'traumatic ulcer']):
        return (6, 'ulcer_lesion')
    
    # Class 8: Calculus (Tartar)
    if any(x in class_name_lower for x in ['calculus', 'tartar', 'dental calculus']):
        return (7, 'calculus_tartar')
    
    # Class 9: Plaque – Heavy
    if any(x in class_name_lower for x in ['plaque', 'dental plaque', 'biofilm']):
        return (8, 'plaque_heavy')
    
    # Class 10: Extrinsic Stain
    if any(x in class_name_lower for x in ['stain', 'discoloration', 'pigmentation', 
                                             'extrinsic', 'black stain']):
        return (9, 'extrinsic_stain')
    
    # Class 11: No Visible Pathology (tooth, healthy)
    if any(x in class_name_lower for x in ['tooth', 'teeth', 'healthy', 'normal', 'sound']):
        # Exclude if it has pathology terms
        if not any(x in class_name_lower for x in ['caries', 'decay', 'cavity', 'diseased']):
            return (10, 'no_visible_pathology')
    
    # Unknown - print warning
    print(f"⚠️  Warning: Could not map class '{class_name}'")
    return None

def convert_coco_to_yolo(coco_file, images_dir, output_dir, split_name):
    """Convert COCO format to YOLO TXT format"""
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get class mapping
    orig_categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Create mapping to 11-class taxonomy
    class_mapping = {}
    class_stats = {}
    
    for orig_id, orig_name in orig_categories.items():
        result = map_class_to_11class(orig_name)
        if result:
            new_id, new_name = result
            class_mapping[orig_id] = new_id
            class_stats[new_name] = class_stats.get(new_name, 0)
    
    # Get image info
    images_info = {img['id']: img for img in coco_data.get('images', [])}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process annotations
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    converted_count = 0
    skipped_count = 0
    
    # Convert each image's annotations
    for img_id, img_info in images_info.items():
        if img_id not in annotations_by_image:
            continue
        
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']
        
        # Create YOLO label file
        label_filename = Path(img_filename).stem + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        yolo_lines = []
        for ann in annotations_by_image[img_id]:
            orig_class_id = ann['category_id']
            
            # Skip if class not mapped
            if orig_class_id not in class_mapping:
                skipped_count += 1
                continue
            
            new_class_id = class_mapping[orig_class_id]
            
            # Get bbox (COCO format: [x, y, width, height])
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            # Ensure values are between 0 and 1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            yolo_lines.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Update stats
            class_name = [k for k, v in class_mapping.items() if v == new_class_id]
            if class_name:
                orig_name = orig_categories[class_name[0]]
                new_name_for_stats = map_class_to_11class(orig_name)[1]
                class_stats[new_name_for_stats] = class_stats.get(new_name_for_stats, 0) + 1
        
        # Write YOLO label file
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted_count += 1
    
    print(f"\n✅ {split_name.upper()}: Converted {converted_count} images")
    print(f"   Skipped {skipped_count} annotations (unmapped classes)")
    
    return class_stats

def copy_images(src_images_dir, dst_images_dir):
    """Copy images to new structure"""
    os.makedirs(dst_images_dir, exist_ok=True)
    
    image_files = list(Path(src_images_dir).glob('*.jpg')) + \
                  list(Path(src_images_dir).glob('*.png')) + \
                  list(Path(src_images_dir).glob('*.jpeg'))
    
    print(f"📁 Copying {len(image_files)} images...")
    for img_file in image_files:
        shutil.copy2(img_file, dst_images_dir)
    
    return len(image_files)

def main():
    print("="*80)
    print("BENCHMARKING DATASET CONVERSION TO COMMON YOLO FORMAT")
    print("="*80)
    
    # Define paths
    root_dir = Path('.')
    output_root = root_dir / 'common_annotation_benchmarking'
    
    # Create output structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    all_class_stats = {}
    
    # Process each split
    for split in splits:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")
        
        # Check which format is available (prefer COCO)
        coco_file = root_dir / split / 'coco' / f'{split}.json'
        yolo_dir = root_dir / split / 'yolo'
        images_dir = root_dir / split / 'images'
        
        output_images = output_root / split / 'images'
        output_labels = output_root / split / 'labels'
        
        if coco_file.exists():
            print(f"📄 Found COCO annotations: {coco_file}")
            
            # Inspect classes first
            class_map = inspect_coco_classes(coco_file)
            
            # Convert COCO to YOLO
            class_stats = convert_coco_to_yolo(
                str(coco_file), 
                str(images_dir),
                str(output_labels),
                split
            )
            
            # Merge stats
            for class_name, count in class_stats.items():
                all_class_stats[class_name] = all_class_stats.get(class_name, 0) + count
            
        elif yolo_dir.exists() and list(Path(yolo_dir).glob('*.txt')):
            print(f"📄 Found YOLO annotations: {yolo_dir}")
            print("⚠️  YOLO format detected - will copy and map classes")
            
            # Copy YOLO files (they should already be in correct format)
            # But we still need to inspect them to see what classes exist
            label_files = list(Path(yolo_dir).glob('*.txt'))
            
            for label_file in label_files:
                shutil.copy2(label_file, output_labels)
            
            print(f"✅ Copied {len(label_files)} YOLO label files")
        else:
            print(f"⚠️  No COCO or YOLO annotations found for {split}")
        
        # Copy images
        if images_dir.exists():
            num_images = copy_images(str(images_dir), str(output_images))
            print(f"✅ Copied {num_images} images")
    
    # Print final statistics
    print("\n" + "="*80)
    print("CONVERSION COMPLETE - CLASS STATISTICS")
    print("="*80)
    
    if all_class_stats:
        print("\nClasses found and converted:")
        for class_name, count in sorted(all_class_stats.items()):
            print(f"  {class_name:30} {count:>6,} annotations")
        print(f"  {'TOTAL':30} {sum(all_class_stats.values()):>6,} annotations")
    
    # Create data.yaml
    create_data_yaml(output_root, all_class_stats)
    
    print("\n✅ Dataset ready at: common_annotation_benchmarking/")
    print("\n📁 Directory structure:")
    print("   common_annotation_benchmarking/")
    print("   ├── train/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   ├── valid/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   ├── test/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   └── data.yaml")

def create_data_yaml(output_root, class_stats):
    """Create data.yaml configuration file"""
    
    # Define all 11 classes in order
    all_classes = [
        'caries_non_cavitated',
        'caries_cavitated',
        'crack_fracture',
        'gross_carious_destruction',
        'gingivitis',
        'abscess_fistula',
        'ulcer_lesion',
        'calculus_tartar',
        'plaque_heavy',
        'extrinsic_stain',
        'no_visible_pathology'
    ]
    
    # Filter to only classes present in this dataset
    present_classes = [cls for cls in all_classes if cls in class_stats]
    
    yaml_content = f"""# Benchmarking Dataset - Common YOLO Format
# Converted: {Path.cwd()}

path: {output_root.absolute()}
train: train/images
val: valid/images
test: test/images

# Number of classes present in this dataset
nc: {len(present_classes)}

# Classes present (subset of 11-class taxonomy)
names:
"""
    
    for i, class_name in enumerate(present_classes):
        yaml_content += f"  {i}: {class_name}\n"
    
    yaml_content += f"""
# Full 11-class taxonomy IDs (for reference):
# 0: caries_non_cavitated
# 1: caries_cavitated
# 2: crack_fracture
# 3: gross_carious_destruction
# 4: gingivitis
# 5: abscess_fistula
# 6: ulcer_lesion
# 7: calculus_tartar
# 8: plaque_heavy
# 9: extrinsic_stain
# 10: no_visible_pathology

# Statistics for this dataset:
"""
    
    for class_name, count in sorted(class_stats.items()):
        yaml_content += f"# {class_name}: {count} annotations\n"
    
    # Write YAML file
    yaml_path = output_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created data.yaml with {len(present_classes)} classes")

if __name__ == "__main__":
    main()
```

---

## USAGE

1. **Extract the downloaded zip file:**
   ```bash
   unzip "Benchmarking Dataset.zip"
   cd "Benchmarking Dataset"
   ```

2. **Run the conversion script:**
   ```bash
   python convert_benchmarking.py
   ```

3. **Expected output directory structure:**
   ```
   common_annotation_benchmarking/
   ├── train/
   │   ├── images/          (5,012 images - JPG/PNG)
   │   └── labels/          (5,012 .txt files - YOLO format)
   ├── valid/
   │   ├── images/          (628 images)
   │   └── labels/          (628 .txt files)
   ├── test/
   │   ├── images/          (629 images)
   │   └── labels/          (629 .txt files)
   └── data.yaml            (YOLOv8 configuration)
   ```

4. **YOLO label format (each .txt file):**
   ```
   <class_id> <x_center> <y_center> <width> <height>
   <class_id> <x_center> <y_center> <width> <height>
   ...
   ```
   
   All coordinates normalized between 0 and 1.

---

## WHAT THE SCRIPT DOES

1. ✅ Reads COCO JSON files from train/valid/test
2. ✅ Inspects and prints all classes found
3. ✅ Maps original classes to 11-class taxonomy
4. ✅ Converts bounding boxes to YOLO format (normalized)
5. ✅ Creates common_annotation_benchmarking/ folder
6. ✅ Copies images to new structure
7. ✅ Generates YOLO label files (one per image)
8. ✅ Creates data.yaml with only classes present
9. ✅ Prints statistics of what was converted

---

## EXPECTED CLASSES IN BENCHMARKING DATASET

Based on the multi-format nature, this dataset likely contains:
- Tooth/Teeth (Class 11)
- Caries/Cavity (Class 2)
- Possibly others (will be revealed when you run the script)

The script will automatically identify and map all classes present.

---

## NOTES

- **Only classes actually present** in the dataset will be in data.yaml
- **Unmapped classes** will be reported as warnings
- **All coordinates** are validated to be between 0-1
- **Image formats** supported: JPG, PNG, JPEG
- **Original data** is preserved (not modified)
