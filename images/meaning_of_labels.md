# MedMNIST Labels

## BloodMNIST

> The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into 8 classes. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images with resolution 3×360×363 pixels are center-cropped into 3×200×200, and then resized into 3×28×28.

| Key  | Value                                                        |
| ---- | ------------------------------------------------------------ |
| 0    | basophil                                                     |
| 1    | eosinophil                                                   |
| 2    | erythroblast                                                 |
| 3    | immature granulocytes (myelocytes, metamyelocytes and promyelocytes) |
| 4    | lymphocyte                                                   |
| 5    | monocyte                                                     |
| 6    | neutrophil                                                   |
| 7    | platelet                                                     |

## DermaMNIST

> The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.

| Key  | Value                                           |
| ---- | ----------------------------------------------- |
| 0    | actinic keratoses and intraepithelial carcinoma |
| 1    | basal cell carcinoma                            |
| 2    | benign keratosis-like lesions                   |
| 3    | dermatofibroma                                  |
| 4    | melanoma                                        |
| 5    | melanocytic nevi                                |
| 6    | vascular lesions                                |

## PathMNIST

> The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.

| Key  | Value                                |
| ---- | ------------------------------------ |
| 0    | adipose                              |
| 1    | background                           |
| 2    | debris                               |
| 3    | lymphocytes                          |
| 4    | mucus                                |
| 5    | smooth muscle                        |
| 6    | normal colon mucosa                  |
| 7    | cancer-associated stroma             |
| 8    | colorectal adenocarcinoma epithelium |

## RetinaMNIST

> The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as the test set. The source images of 3×1,736×1,824 are center-cropped and resized into 3×28×28.

| Key  | Value |
| ---- | ----- |
| 0    | 0     |
| 1    | 1     |
| 2    | 2     |
| 3    | 3     |
| 4    | 4     |

