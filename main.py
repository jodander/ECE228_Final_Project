import numpy as np
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm as svm_sk
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

def main():

	# Preprocessing Training data
	fruit_images_t = []
	labels_t = [] 
	for fruit_dir_path in glob.glob("fruits-360/Training/*"):
		fruit_label = fruit_dir_path.split("\\")[-1]
		for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			
			image = cv2.resize(image, (45, 45))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
			fruit_images_t.append(image)
			labels_t.append(fruit_label)
	fruit_images_t = np.array(fruit_images_t)
	labels_t = np.array(labels_t)

	# Create dicts/arrays for Training Data
	names_t = np.unique(labels_t)
	ids_t = [k for k in range(0, len(names_t))]
	name_id_t = list(zip(names_t, ids_t))
	id_to_name_t = {id: name for (name, id) in name_id_t}
	name_to_id_t = {name: id for (name, id) in name_id_t}
	label_ids_t = np.array([name_to_id_t[x] for x in labels_t])
	
	# Scale Training data and then run PCA on it
	scaler = StandardScaler()
	scaled_images = scaler.fit_transform([im.flatten() for im in fruit_images_t])
	pca = PCA(n_components=80)
	pca_result_t = pca.fit_transform(scaled_images)
	
	# Split training set
	X_train, X_test, y_train, y_test = train_test_split(pca_result_t, label_ids_t, test_size=0.6)
	
	# Train SVM
	svm = svm_sk.SVC()
	svm = svm.fit(X_train, y_train)
	
	# Train Random Forest
	forest = RandomForestClassifier(n_estimators=10)
	forest = forest.fit(X_train, y_train)
	
	# Test both classifiers on X_test if instructed
	if len(sys.argv) == 1 or sys.argv[1] == 1:
		# Make predictions w/ both classifiers
		forest_test_predictions = forest.predict(X_test)
		svm_test_predictions = svm.predict(X_test)
		
		# Compute accuracy scores
		svm_precision_t = accuracy_score(svm_test_predictions, y_test) * 100
		forest_precision_t = accuracy_score(forest_test_predictions, y_test) * 100 
		print("Accuracy with SVM on X_test: {0:.6f}".format(svm_precision_t))
		print("Accuracy with RF on X_test: {0:.6f}".format(forest_precision_t))
	
		# Compute Confusion Matrices for both classifiers
		cm_svm_t = confusion_matrix(y_test, svm_test_predictions)
		cm_forest_t = confusion_matrix(y_test, forest_test_predictions)
		cm_svm_t = normalize(cm_svm_t, axis=0, norm='l1')
		cm_forest_t = normalize(cm_forest_t, axis=0, norm='l1')
		
		# Plot both confusion matrices (separate windows)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm_svm_t)
		plt.title('Confusion matrix for SVM on X_test')
		fig.colorbar(cax)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.show()
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm_forest_t)
		plt.title('Confusion matrix for Random Forest on X_test')
		fig.colorbar(cax)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.show()
		
	
	if len(sys.argv) > 1 and sys[1] == 0:
		return
	
	# Preprocessing Validation data
	fruit_images_v = []
	labels_v = [] 
	for fruit_dir_path in glob.glob("fruits-360/Validation/*"):
		fruit_label = fruit_dir_path.split("\\")[-1]
		for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			
			image = cv2.resize(image, (45, 45))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
			fruit_images_v.append(image)
			labels_v.append(fruit_label)
	fruit_images_v = np.array(fruit_images_v)
	labels_v = np.array(labels_v)
	
	# Create dicts/arrays for Validation data
	names_v = np.unique(labels_v)
	ids_v = [k for k in range(len(names_v))]
	id_name_v = list(zip(ids_v, names_v))
	name_to_id_v = {name: id for (id, name) in id_name_v}
	id_to_name_v = {id: name for (id, name) in id_name_v}
	label_ids_v = np.array([name_to_id_v[x] for x in labels_v])
	
	# Scale Validation Data and run PCA on it
	images_scaled = scaler.transform([im.flatten() for im in fruit_images_v])
	pca_result_v = pca.transform(images_scaled)
	
	# Predict on with all classifiers Validation data
	svm_prediction_v = svm.predict(pca_result_v)
	forest_prediction_v = forest.predict(pca_result_v)
	
	# Compute accuracy scores for all classifiers
	svm_precision_v = accuracy_score(svm_prediction_v, label_ids_v) * 100
	forest_precision_v = accuracy_score(forest_prediction_v, label_ids_v) * 100
	
	print("Accuracy with SVM on Validation data: {0:.6f}".format(svm_precision_v))
	print("Accuracy with RF on Validation data: {0:.6f}".format(forest_precision_v))
	
	# Compute confusion matrices for all classifiers
	cm_svm_v = confusion_matrix(label_ids_v, svm_prediction_v)
	cm_svm_v = normalize(cm_svm_v, axis=0, norm='l1')
	
	cm_forest_v = confusion_matrix(label_ids_v, forest_prediction_v)
	cm_forest_v = normalize(cm_forest_v, axis=0, norm='l1')
	
	# Plot both confusion matrices (separate windows)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm_svm_v)
	plt.title('Confusion matrix for SVM on Validation data')
	fig.colorbar(cax)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm_forest_v)
	plt.title('Confusion matrix for Random Forest on Validation data')
	fig.colorbar(cax)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()

main()