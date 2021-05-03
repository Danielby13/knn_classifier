# OCR of Handwritten Hebrew 
This project was built as part of Image processing and Computer vision academic course.
We used k-Nearest Neighbor algorithm to classify handwriting images, using a database of about 5000 images.

## Getting Started
To run the program,first you need to clone the project.
and then in terminal:
```
cd to clone file is placed
python knn_classifier.py [path of hhd_dataset]
```
When the program ends:
	the program will export csv and text file where "knn_classifier.py" is placed.
	the csv contain confusion matrix
	the text file contain the accuracy for each letter and the best k and distance function (euclidean or chi2 square)

## Authors:
*Daniel Ben Yair
*Inbal Altan

### References
[I. Rabaev, B. Kurar Barakat, A. Churkin and J. El-Sana. The HHD Dataset. The 17th
International Conference on Frontiers in Handwriting Recognition, pp. 228-233, 2020](https://www.researchgate.net/publication/343880780_The_HHD_Dataset)