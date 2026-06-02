---
tags: [project/captura, ai-service, dataset, research, computer-vision]
date: 2026-06-02
status: research-report
parent: [[PRD|Captura AI Service PRD]]
---

# Dataset Research for Captura AI Service

Generated: 2026-06-02  
Goal: Identify datasets that can support Captura AI image analysis for vehicle type, vehicle color, license plate, and supporting visual parameters.

## 1. Executive Summary

Captura should not rely on a single dataset. The recommended MVP path is a modular pipeline:

1. Use a strong pretrained detector for common objects and vehicles.
2. Fine-tune vehicle type detection/classification with traffic-scene datasets.
3. Train vehicle color classification on cropped vehicle datasets.
4. Train or fine-tune Indonesian license plate detection/OCR separately.
5. Use CLIP-style semantic embeddings for flexible attributes such as clothing, style, and scene descriptors.

Best starting stack:

- Vehicle type: MIO-TCD + BDD100K + COCO/Open Images.
- Vehicle color: VCoR + UFPR-VCR.
- Indonesian plate: Indonesian License Plate Dataset from Kaggle or Zenodo, with license review before production.
- Global plate robustness: UFPR-ALPR.
- Supporting person/clothing attributes: Market-1501 Attribute, PETA, DeepFashion, and CLIP zero-shot prompts.

## 2. Captura Requirements Mapping

| Captura parameter | Dataset need | Recommended approach |
| --- | --- | --- |
| Vehicle type | car, motorcycle, bicycle, bus, truck, van, pickup | Detector or vehicle crop classifier |
| Vehicle color | black, white, silver, red, blue, etc. | Classifier on vehicle crop after detection |
| Plate number | plate box and characters | Separate plate detector + OCR/character recognizer |
| Location/time | EXIF/manual/backend metadata | Not primarily AI dataset-driven |
| Clothing/style | jacket color, helmet, bag, outfit style | CLIP + pedestrian/clothing attribute datasets |
| Scene context | street, rain, night, golden hour, road | CLIP tags + street-scene datasets |

## 3. Highest-Priority Datasets

## 3.1 MIO-TCD

Best for: vehicle type classification and localization.

Why it matters:

- MIO-TCD classification contains 648,959 images in 11 categories, including articulated truck, bicycle, bus, car, motorcycle, pickup truck, single unit truck, work van, pedestrian, non-motorized vehicle, and background.
- MIO-TCD localization contains 137,743 high-resolution images with foreground object labels.
- License is Creative Commons Attribution-NonCommercial-ShareAlike 4.0, so it is excellent for research/prototype and needs review for commercial production.

Recommended Captura use:

- Use as a first fine-tuning dataset for vehicle type classifier.
- Use localization data to improve detection for vehicle subclasses beyond basic COCO classes.
- Map labels into Captura taxonomy:
  - `car`
  - `motorcycle`
  - `bicycle`
  - `bus`
  - `truck`
  - `pickup`
  - `van`
  - `other_vehicle`

Source: [MIO-TCD official dataset page](https://tcd.miovision.com/challenge/dataset.html).

## 3.2 BDD100K

Best for: street/driving scene detection with vehicle classes.

Why it matters:

- BDD100K is a large driving dataset designed for heterogeneous multitask learning.
- It is commonly used for detection tasks involving car, bus, truck, motorcycle, bicycle, person/rider, traffic light, and traffic sign.
- It is closer to road/street imagery than clean product photos.

Recommended Captura use:

- Use for robust detection in varied lighting, road, and street conditions.
- Useful for Captura's street-photo search because images may include real traffic context.
- Combine with MIO-TCD if Captura needs better subclassing for pickup/van/truck.

Source: [BDD100K paper](https://arxiv.org/abs/1805.04687), [BDD100K class references via Vis4D](https://vis4d.readthedocs.io/en/latest/_modules/vis4d/data/datasets/bdd100k.html), [BDD100K Kaggle mirror](https://www.kaggle.com/datasets/alvaromalfaro/bdd100k).

## 3.3 VCoR Vehicle Color Recognition Dataset

Best for: vehicle color classification.

Why it matters:

- VCoR contains 10k+ vehicle images.
- It has 15 color classes: white, black, grey, silver, red, blue, brown, green, beige, orange, gold, yellow, purple, pink, and tan.
- It is specifically designed for vehicle color recognition rather than general object detection.

Recommended Captura use:

- Train a color classifier on vehicle crops produced by YOLO/vehicle detector.
- Start with broad UI buckets:
  - black
  - white
  - silver/gray
  - red
  - blue
  - green
  - yellow/gold
  - brown/beige/tan
  - other/unknown

Production caution:

- Kaggle lists "Data files © Original Authors"; verify license terms before production/commercial use.

Source: [VCoR Kaggle dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset).

## 3.4 UFPR-VCR

Best for: vehicle color recognition in real-world conditions.

Why it matters:

- UFPR-VCR is described as a vehicle color recognition dataset with 10,039 images.
- It includes challenging real-world conditions such as frontal/rear views, partial occlusions, lighting variations, and nighttime scenes.

Recommended Captura use:

- Use to improve robustness after VCoR baseline.
- Particularly relevant for street photos where vehicle color is affected by shadow, night, blur, and partial occlusion.

Source: [UFPR-VCR GitHub repository](https://github.com/lima001/ufpr-vcr-dataset).

## 3.5 Indonesian License Plate Dataset

Best for: Indonesian plate detection and recognition.

Why it matters:

- The Kaggle dataset is specifically designed for Indonesian license plate detection and recognition.
- It includes 1,906 license plates with varied orientations, blur, and environmental conditions.
- It includes a recognition dataset with cropped plates and character annotations for 0-9 and A-Z in YOLO format.

Recommended Captura use:

- Use for prototype plate detection and character recognition.
- Use as initial local benchmark for Indonesian plate OCR.
- Combine with EasyOCR/PaddleOCR post-processing and Indonesian plate regex validation.

Production caution:

- Kaggle lists license as unknown. Treat as research/prototype only until usage rights are clarified.

Source: [Indonesian License Plate Dataset on Kaggle](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset).

## 3.6 Indonesian License Plate Detection Dataset on Zenodo

Best for: alternative Indonesian plate detection source.

Why it matters:

- Zenodo record describes an Indonesian vehicle license plate detection dataset intended for object detection and exported in YOLOv11 format.
- Zenodo records often provide clearer archival metadata than ad-hoc mirrors, but license must still be checked per record.

Recommended Captura use:

- Use as a second Indonesian plate detection source if license permits.
- Compare with Kaggle Indonesian plate dataset for coverage and annotation consistency.

Source: [Zenodo Indonesian License Plate Detection Dataset](https://zenodo.org/records/15605718).

## 3.7 UFPR-ALPR

Best for: robust ALPR pipeline benchmark.

Why it matters:

- UFPR-ALPR includes 4,500 fully annotated images acquired in real-world scenarios where both vehicle and camera are moving.
- It is not Indonesian, but it is useful for plate detection/OCR robustness and ALPR benchmarking.

Recommended Captura use:

- Use as auxiliary training/evaluation for plate detector robustness.
- Do not rely on it for Indonesian character/layout specifics.

Source: [UFPR-ALPR GitHub repository](https://github.com/raysonlaroca/ufpr-alpr-dataset), [UFPR-ALPR paper](https://arxiv.org/abs/1802.09567).

## 4. Additional Datasets Worth Considering

## 4.1 COCO

Best for: general object detection baseline.

Useful labels:

- person
- bicycle
- car
- motorcycle
- bus
- truck
- backpack
- handbag
- umbrella

Recommended Captura use:

- Keep COCO-pretrained YOLO as the baseline detector.
- Use COCO for accessories and broad object context.
- Do not expect fine-grained vehicle type or color labels.

Source: [COCO official site](https://cocodataset.org/index.htm), [COCO category reference](https://aiwiki.ai/wiki/coco_dataset).

## 4.2 Open Images V7

Best for: large-scale general object detection.

Why it matters:

- Open Images V7 is large and includes many relevant object classes.
- Ultralytics references vehicle-related class names such as Bicycle, Bus, Car, Land vehicle, and Motorcycle.

Recommended Captura use:

- Use for broader object detection pretraining or class expansion.
- Good for generalization, but dataset size and complexity are high.

Source: [Open Images V7 via Ultralytics docs](https://docs.ultralytics.com/datasets/detect/open-images-v7), [Open Images V7 via FiftyOne docs](https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html).

## 4.3 nuImages

Best for: street/driving object detection and taxonomy.

Why it matters:

- nuImages taxonomy includes categories such as `vehicle.car`, `vehicle.truck`, and cycle-related categories.
- It is suitable for autonomous driving-style image understanding.

Recommended Captura use:

- Use if Captura needs another high-quality street/driving dataset.
- More useful for detection/scene understanding than vehicle color.

Source: [nuImages tutorial and taxonomy reference](https://www.nuscenes.org/public/tutorials/nuimages_tutorial.html).

## 4.4 Mapillary Vistas

Best for: street-scene segmentation.

Why it matters:

- Mapillary Vistas is a street-level scene understanding dataset with pixel-level annotations and vehicle classes such as bus, car, and motorcycle in its taxonomy.
- Mapillary imagery is geospatial and street-level, close to Captura's public-space context.

Recommended Captura use:

- Use for scene segmentation or road/street context.
- Less direct for vehicle color classification.

Source: [Mapillary Vistas ICCV supplemental](https://openaccess.thecvf.com/content_ICCV_2017/supplemental/Neuhold_The_Mapillary_Vistas_ICCV_2017_supplemental.pdf), [Mapillary data documentation](https://help.mapillary.com/hc/en-us/articles/360003021152-Types-of-map-data).

## 4.5 VeRi-776

Best for: vehicle attributes and re-identification experiments.

Why it matters:

- Literature describes VeRi/VeRi-776 as a real-world urban surveillance vehicle ReID dataset with labels including bounding boxes, types, colors, and brands.
- It contains around 49k-50k images of 776 vehicles from 20 cameras.

Recommended Captura use:

- Consider for vehicle attribute experiments: color, type, and re-identification-like retrieval.
- Check access and annotation availability before committing.

Source: [VeRi-776 Papers With Code](https://paperswithcode.com/dataset/veri-776), [V2ReID article discussing VeRi attributes](https://pmc.ncbi.nlm.nih.gov/articles/PMC9692519/).

## 4.6 Market-1501 Attribute / PETA / DeepFashion

Best for: person clothing and accessory attributes.

Why it matters:

- Market-1501 Attribute includes clothing color, bag/accessory, hat, gender/age, sleeve/lower clothing attributes in the person ReID context.
- PETA includes pedestrian appearance, hair, clothing, and accessory attributes.
- DeepFashion is useful for clothing categories and fine-grained attributes, but less street-contextual.

Recommended Captura use:

- Use these only for supporting "person appearance" search, not for identity recognition.
- Combine with CLIP zero-shot prompts rather than building face or identity recognition.

Privacy caution:

- Captura should avoid face recognition or personal identity matching in MVP.

Sources: [Human Attribute Recognition survey](https://www.mdpi.com/2076-3417/10/16/5608), [Market1501-Attributes Papers With Code](https://paperswithcode.com/dataset/market1501-attributes), [DeepFashion attribute paper](https://arxiv.org/abs/1807.11674).

## 5. Recommended MVP Dataset Plan

## Phase 1: Baseline Without Heavy Training

Use:

- Existing YOLOv8 pretrained detector.
- EasyOCR with Indonesian plate regex.
- CLIP ViT-B/32 for visual tags and embeddings.

Goal:

- Validate Captura product flow first.
- Collect real user/fotografer data as soon as possible.

## Phase 2: Vehicle Type Fine-Tuning

Use:

- MIO-TCD for vehicle type classification/localization.
- BDD100K for street-condition robustness.
- COCO/Open Images as broad detector pretraining/reference.

Target labels:

- `car`
- `motorcycle`
- `bicycle`
- `bus`
- `truck`
- `pickup`
- `van`
- `other_vehicle`

## Phase 3: Vehicle Color Classifier

Use:

- VCoR.
- UFPR-VCR.
- Captura-collected vehicle crops from Indonesian street photographers.

Target labels:

- `black`
- `white`
- `silver_gray`
- `red`
- `blue`
- `green`
- `yellow_gold`
- `brown_beige_tan`
- `orange`
- `other`
- `unknown`

## Phase 4: Indonesian License Plate Improvement

Use:

- Indonesian License Plate Dataset from Kaggle for prototype.
- Zenodo Indonesian plate dataset if license permits.
- UFPR-ALPR as robustness benchmark.
- Captura's own consented/curated Indonesian plate samples.

Pipeline:

1. Detect plate box.
2. Crop and preprocess.
3. OCR candidate text.
4. Normalize text.
5. Validate with Indonesian regex.
6. Store candidates and confidence, not only one final string.

## Phase 5: Supporting Visual Attributes

Use:

- CLIP prompts for initial attributes.
- Market-1501 Attribute/PETA for person clothing/accessory experiments.
- DeepFashion for clothing taxonomy experiments.

Captura should initially expose broad attributes only:

- Clothing color.
- Helmet/no helmet.
- Bag/backpack.
- Runner/cyclist/motor rider.
- Broad style descriptors.

## 6. Recommended Model Architecture

## 6.1 Multi-Model Pipeline

Recommended pipeline for `ai-service`:

1. Object detector detects person, vehicle, motorcycle, bicycle, car, bus, truck, plate if available.
2. Vehicle crop is passed to:
   - vehicle type classifier.
   - vehicle color classifier.
3. Plate crop is passed to:
   - OCR.
   - regex normalizer.
4. Full image and relevant crops are passed to:
   - CLIP embedding.
   - zero-shot visual tag prompts.
5. Backend stores structured metadata and embedding.

## 6.2 Why Not One Giant Model?

Captura's requirements are multi-attribute and partially sensitive. A modular pipeline is easier to debug:

- Vehicle type can improve without touching OCR.
- Plate OCR can use Indonesia-specific data.
- Color classifier can run only on vehicle crops.
- CLIP can handle flexible text search while supervised datasets mature.

## 7. License and Production Risk

| Dataset | Production risk | Notes |
| --- | --- | --- |
| MIO-TCD | Medium/high | CC BY-NC-SA; non-commercial restriction needs review |
| BDD100K | Medium | Review official license before commercial use |
| VCoR | Medium/high | Kaggle says data files owned by original authors |
| UFPR-VCR | Medium | Check repository license and citation requirements |
| Indonesian License Plate Kaggle | High | License unknown |
| Zenodo Indonesian Plate | Medium | Check exact Zenodo record license |
| UFPR-ALPR | Medium | Good research benchmark; check license |
| COCO | Lower | Commonly used; still verify license and attribution |
| Open Images | Lower/medium | Large and useful; check image-level licenses |
| Market/PETA/DeepFashion | Medium | Research use likely; check redistribution and commercial terms |

Recommendation:

- Use public datasets for research and prototype.
- Build a Captura-owned dataset as early as possible from photographer uploads with clear contributor terms.
- Store consent/license metadata for all training samples.

## 8. Practical Next Steps

1. Create `ai-service/data/datasets_manifest.yml` to track dataset name, source URL, license, intended use, and local path.
2. Start with pretrained YOLOv8 + CLIP + EasyOCR to validate current PRD flows.
3. Add vehicle crop export from existing analysis pipeline.
4. Download and normalize MIO-TCD for vehicle type experiments.
5. Download and normalize VCoR or UFPR-VCR for color experiments.
6. Build a small Indonesian validation set from real Captura-like images.
7. Add evaluation metrics:
   - vehicle type accuracy.
   - vehicle color accuracy.
   - plate detection mAP.
   - OCR character accuracy.
   - full-plate exact/partial match.
   - semantic retrieval top-k relevance.

## 9. Source List

1. [MIO-TCD official dataset page](https://tcd.miovision.com/challenge/dataset.html) - vehicle classification/localization dataset with 11 classes.
2. [BDD100K paper](https://arxiv.org/abs/1805.04687) - large driving dataset for multitask learning.
3. [BDD100K class reference via Vis4D](https://vis4d.readthedocs.io/en/latest/_modules/vis4d/data/datasets/bdd100k.html) - confirms common vehicle classes.
4. [VCoR Kaggle dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset) - vehicle color dataset with 15 color classes.
5. [UFPR-VCR GitHub repository](https://github.com/lima001/ufpr-vcr-dataset) - vehicle color recognition dataset in real-world conditions.
6. [Indonesian License Plate Dataset on Kaggle](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset) - Indonesian plate detection and recognition dataset.
7. [Zenodo Indonesian License Plate Detection Dataset](https://zenodo.org/records/15605718) - Indonesian plate detection dataset in YOLO format.
8. [UFPR-ALPR GitHub repository](https://github.com/raysonlaroca/ufpr-alpr-dataset) - ALPR dataset with moving vehicle/camera conditions.
9. [COCO official site](https://cocodataset.org/index.htm) - general object detection dataset.
10. [Open Images V7 via Ultralytics docs](https://docs.ultralytics.com/datasets/detect/open-images-v7) - large object detection dataset with many vehicle-related classes.
11. [nuImages tutorial](https://www.nuscenes.org/public/tutorials/nuimages_tutorial.html) - street/driving dataset taxonomy reference.
12. [Mapillary Vistas supplemental](https://openaccess.thecvf.com/content_ICCV_2017/supplemental/Neuhold_The_Mapillary_Vistas_ICCV_2017_supplemental.pdf) - street-scene segmentation taxonomy.
13. [VeRi-776 Papers With Code](https://paperswithcode.com/dataset/veri-776) - vehicle re-identification dataset summary.
14. [Human Attribute Recognition survey](https://www.mdpi.com/2076-3417/10/16/5608) - overview of pedestrian/person attribute datasets including PETA and Market-1501 Attribute.
15. [Market1501-Attributes Papers With Code](https://paperswithcode.com/dataset/market1501-attributes) - person attribute dataset summary.
16. [DeepFashion attribute paper](https://arxiv.org/abs/1807.11674) - clothing category and fine-grained attribute reference.

## 10. Research Methodology

Searched current web sources for vehicle type, vehicle color, Indonesian license plate, global ALPR, and person/clothing attribute datasets. Prioritized official dataset pages, academic papers, GitHub repositories, and established dataset indexes. Dataset licenses were treated conservatively when not clearly stated.

