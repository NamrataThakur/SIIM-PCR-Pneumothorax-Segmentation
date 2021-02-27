# SIIM-PCR-Pneumothorax-Segmentation

Business Problem:

Pneumothorax is a medical condition which arises when air leaks into the space between the lung and the chest wall. This air pushes on the outside of the lung and makes it collapse. Thus, pneumothorax can be a complete lung collapse or a portion of the lungs may be collapsed. Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or sometimes for no obvious reason at all. It can be a life-threatening event.
Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. The problem that this case study is dealing with predicts whether the condition exists in the chest x-ray image given and if present it segments the portion of the lungs that is affected. An accurate prediction would be useful in a lot of clinical scenarios to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

Mapping the real world problem as a Deep Learning problem:

The data is comprised of images in DICOM format containing run-length-encoded (RLE) masks. The instances of pneumothorax are indicated by encoded binary masks in the annotations. Some training images have multiple annotations depicting the multiple locations of the event. Images without pneumothorax have a mask value of -1. The task is to predict the mask of pneumothorax in the given X-ray image. This task can be mapped as a Semantic Image Segmentation problem.

Data set Analysis:

•	Files given:  train-rle.csv, stage_2_sample_submission.csv (test_data), train_images, test_images.
•	Total File Size : 4GB
•	Total number of records: 12,954 (train_data), 3204 (test_data)
•	The train-rle.csv contains image IDs and their corresponding RLE masks and the test csv file only contains the image IDs.

Real World Business Constraints: 

•	Low latency is important.
•	Mis-classification/ mis-segmentation cost is considerably high as we are dealing with medical data and thus it is very sensitive to such errors.

Performance Metric:

Metric(s):

•	Dice Coefficient  (IntersectionOverUnion/IOU)
•	Combo Loss – (Binary Cross Entropy + Dice Loss/ F1 loss)
•	Confusion Matrix


THIS REPO IS WORK IN PROGRESS. NEW ADDITION/UPDATION IS DONE EVERYDAY.
