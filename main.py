import numpy as np
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def main():

	# Preprocessing
	fruit_images = []
	labels = [] 
	for fruit_dir_path in glob.glob("fruits-360/Training/*"):
		fruit_label = fruit_dir_path.split("\\")[-1]
		for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			
			image = cv2.resize(image, (45, 45))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
			fruit_images.append(image)
			labels.append(fruit_label)
	fruit_images = np.array(fruit_images)
	labels = np.array(labels)

	# Create dicts
	name = np.unique(labels)
	ids = [k for k in range(0, len(name))]
	name_id = list(zip(name, ids))
	id_to_name = {id: name for (name, id) in name_id}
	name_to_id = {name: id for (name, id) in name_id}
	
	#plt.figure()
	#plt.imshow(fruit_images[203])
	#plt.show()
	
	scaler = StandardScaler()
	scaled_images = scaler.fit_transform([im.flatten() for im in fruit_images])
	
	pca = PCA(n_components=50)
	pca_result = pca.fit_transform(scaled_images)
	
	#labels = np.array(ids).reshape(len(ids),1)
	X_train, X_test, y_train, y_test = train_test_split(pca_result, labels, test_size=0.25, random_state=42)
	
	svm_clf = svm.SVC()
	svm_clf = svm_clf.fit(X_train, y_train)

	test_predictions = svm_clf.predict(X_test)
	precision = accuracy_score(test_predictions, y_test) * 100
	print("Accuracy with SVM: {0:.6f}".format(precision))
	
main()