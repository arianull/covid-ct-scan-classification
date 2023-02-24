# COVID-19 Lung CT Scan Image Data Classification using Machine Learning, CNN, Transfer Learning
CNN Transfer learning, SVM, Logistic Regression on Covid-19 CT Scan images.

This project is focused on applying machine learning algorithms to COVID-19 lung CT scan image data. Specifically, three different algorithms were utilized for analysis: Support Vector Machines (SVM), Logistic Regression, and Convolutional Neural Networks (CNN) with transfer learning using the VGG16 model. The goal of this project is to explore the efficacy of different machine learning techniques in accurately identifying COVID-19 lung CT scans.

### Dataset
The dataset used in this project is the COVID-19 CT Lung and Infection Segmentation Dataset. The non-COVID-19 scans include scans from patients with other types of lung infections as well as scans from healthy patients.

![alt text](#?raw=true)

### Algorithms
Support Vector Machines (SVM)
SVM is a supervised machine learning algorithm that is commonly used for classification problems. SVM tries to find the optimal hyperplane that separates data points into different classes. In this project, SVM was used to classify COVID-19 and non-COVID-19 lung CT scans. The SVM algorithm was implemented using the scikit-learn library.

### Logistic Regression
Logistic regression is another supervised machine learning algorithm that is commonly used for binary classification problems. In this project, logistic regression was used to classify COVID-19 and non-COVID-19 lung CT scans. The logistic regression algorithm was implemented using the scikit-learn library.

### Convolutional Neural Networks (CNN) with transfer learning using the VGG16 model
CNNs are a type of neural network that are commonly used for image classification tasks. Transfer learning involves using pre-trained models to improve performance on a new task. In this project, the VGG16 model, which was pre-trained on the ImageNet dataset, was used for transfer learning. The last layer of the VGG16 model was replaced with a new fully connected layer, and the model was fine-tuned on the COVID-19 CT Lung and Infection Segmentation Dataset.

### Results
The results of the analysis are presented in a Jupyter notebook in this repository. Overall, the CNN with transfer learning using the VGG16 model achieved the highest accuracy, with an accuracy of 99%. The SVM algorithm achieved an accuracy of 92%, and the logistic regression algorithm achieved an accuracy of 89%. These results suggest that CNNs with transfer learning can be an effective tool for identifying COVID-19 lung CT scans.

### Conclusion
In conclusion, this project demonstrates the efficacy of different machine learning techniques in accurately identifying COVID-19 lung CT scans. The results suggest that CNNs with transfer learning can be particularly effective for this task. However, further research is needed to validate these findings and to explore the potential clinical applications of machine learning algorithms in COVID-19 diagnosis and treatment.
