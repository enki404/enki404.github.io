

## Drainedimages folder contains all images ##
## 1:e, 2:density D/M/L(0,1,2), 3: p, 4: q, 5:epsla
## 6: epslv, 7: sigma1, 8:sigma2, 9:sigma3 10: CN

import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from sklearn.model_selection import train_test_split



### Load Dataset ###

path_dir =r'E:\9.18.22\CSR=0.2\DrainedImages'


image_paths = []
void_ratios = []
densitys = []

for imgfile in os.listdir(path_dir):
    image_path = os.path.join(path_dir,imgfile)
    void_ratio = imgfile.split('_')[0]
    density = imgfile.split('_')[1]  ## this is 'str', np.array() can use to convert str to append
    image_paths.append(image_path)
    void_ratios.append(void_ratio)
    densitys.append(density)
    
    
### Convert to DF ###

df = pd.DataFrame()
df['Image'],df['Void'],df['Density']= image_paths, void_ratios, densitys
# df.head()

### Labels for Density ###

density_dict = {0:'Dense',1:'Medium',2:'Loose'}

y_label = np.array(df['Density'].astype(int)).T

actual_y_label = y_label

### OneHot Encoded Label for y ###
# number of labels: how many types of label in raw dataset
# temp = y_label
# # number of rows in dataset
# n = np.size(df, 0)
# num of categrical class
# numoflabel = 3
# onehot_labels = np.zeros((n,numoflabel))

# for i in range(n):
#     onehot_labels[i,temp[i]-1] = 1

# y_label = onehot_labels
# print("Actual One Hot Encoded y label is:\n", y_label, "\n", "Shape of actual y label:\n", y_label.shape)



### Exploratory Image Data Analysis ###

#img = cv2.imread(df['Image'][0])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.axis('off')
#plt.imshow(img);


### resize/grayscale imgs for better preprocessing ###
### Feature Extraction ###

def extract_feature(images):
    features=[]
    for image in images:
        grayscale_img = load_img(image,color_mode='grayscale',target_size =(128,128),keep_aspect_ratio=True)     
        new_img = np.array(grayscale_img)
        features.append(new_img)
        
    features = np.array(features)
    
    # 1 represent grayscale, RGB color img = 3
    features = features.reshape(len(features),128,128,1)
        
    return features

### Input Data Marix ###

X = extract_feature(df['Image'])

#X.shape=(1224,128,128,1)=length of features, width,height,grayscale

### Display part(grid) of images ###
plt.figure(figsize=(10,10))

img_grid1 = df.iloc[0:3]
img_grid2 = df.iloc[660:663]
img_grid3 = df.iloc[1070:1073]

img_grid = pd.concat([img_grid1,img_grid2,img_grid3], axis=0, ignore_index=True)

## iterate over DF rows as namedtuples
for index, image, void, state in img_grid.itertuples():
    plt.subplot(3,3,index+1)
    img = np.array(load_img(image))
    void = float(void)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Void Ratio: {str(round(void,3))} \n  State: {density_dict[int(state)]}',fontsize = 10)

### Normalizatio of Input Data ###
## pixel values are in range of 0-255
X = X/255.0

y_void = df['Void'].values.astype(float)
y_density = df['Density'].values.astype(float)

### determine the class/dimension of input and output ###

input_shape = (128,128,1)
output_shape = 1

### Building NN Model ###
## tf.keras.Input()
## shape not concluding batch size, the expected input will be batches of n-dimensional vectors. Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.
## filters: integer,dimensionality of output space, i.e. the num of output filters in conv
## kenrel_size: width and height of 2d conv window
## strides: strides of the convolution along the height and width

# Input(shape=(len(train.columns)
inputs = Input((input_shape),name='Input')
print("Inputs:\n", inputs)

#@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
#def l1_reg(weight_matrix):
#   return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_matrix))


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True)
### Convolutional Layers ###

conv1 = Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same')(inputs)
batch1 = tf.keras.layers.BatchNormalization(axis=-1,
    momentum=0.99,
    epsilon=0.009,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones')(conv1)
maxp1 = MaxPooling2D(pool_size=(3,3))(batch1)

conv2 = Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same')(maxp1)
batch2 = tf.keras.layers.BatchNormalization(axis=-1,
    momentum=0.99,
    epsilon=0.15,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones')(conv2)
maxp2 = MaxPooling2D(pool_size=(3,3))(batch2)

conv3 = Conv2D(128,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(maxp2)
bacth3 = tf.keras.layers.BatchNormalization()(conv3)
maxp3 = MaxPooling2D(pool_size=(2,2))(conv3)

conv4 = Conv2D(64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(maxp3)
batch4 = tf.keras.layers.BatchNormalization()(conv4)
maxp4 = MaxPooling2D(pool_size=(2,2))(conv4)

## conv layer have the weights in terms of matrix-like strucure

flatten = Flatten()(maxp4)

### Fully Connected layer ###

fc1 = Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.09))(flatten)
batch5 = tf.keras.layers.BatchNormalization()(fc1)
dropout1 = Dropout(0.5)(batch5)
fc2 = Dense(32,activation='relu')(flatten)

dropout11 = Dropout(0.6)(dropout1)
dropout2 = Dropout(0.05)(fc2)

## regression/classificcation

output1 = Dense(3,activation='softmax',name='density')(dropout11)
output2 = Dense(1,activation='linear',name='void')(dropout2)



model = Model(inputs = [inputs], outputs = [output1,output2])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# def custom_loss(y_true,y_pred):
#     mae = tf.keras.losses.MeanSquaredError()
#     return mae

model.compile(loss={'density':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),'void':tf.keras.losses.MeanSquaredError()} ,optimizer=opt, metrics = {'density':'accuracy','void':'mse'})

print("Model Summary:\n",model.summary())

### Plot Model ###

from tensorflow.keras.utils import plot_model
plot_model(model,to_file=r'E:\9.18.22\model.png')


from sklearn.model_selection import train_test_split

X_train, X_test, y_void_train, y_void_test, y_density_train, y_density_test = train_test_split(X, y_void, y_label, test_size=0.2, shuffle= True,random_state=233)
X_train,X_val,y_void_train,y_void_val,y_density_train,y_density_val = train_test_split(X_train,y_void_train,y_density_train, test_size = 0.1, shuffle= True, random_state=233)

# x_norm = (b-a)*(x-min/max-min)+a
# scale to 0 ~100


### Train Model ###
## verbose =0, show nothing, =1 progress bar, =2 just epoch i/total
## callbacks=callback
history = model.fit(x=X_train,y=[y_density_train,y_void_train],batch_size=32,epochs=300,validation_data=(X_val,[y_density_val,y_void_val]))


### Loss & Accuracy Results ###

def PercentError(y_actual,y_expected):
    E = abs((y_actual-y_expected)/y_expected)
    return E

plt.figure()

density_acc = history.history['density_accuracy']
val_density_acc = history.history['val_density_accuracy']

epoch_density = range(len(density_acc))

plt.plot(epoch_density,density_acc,'b',label='Training Accuracy')
plt.plot(epoch_density,val_density_acc,'r',label='Validation Accuracy')
plt.title('Accuracy of Density')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


density_loss = history.history['density_loss']
val_density_loss = history.history['val_density_loss']


plt.plot(epoch_density,density_loss,'b',label='Training Loss')
plt.plot(epoch_density,val_density_loss,'r',label='Validation Loss')
plt.title('Loss of Density')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


void_acc = history.history['void_mse']
val_void_acc = history.history['val_void_mse']

epoch_void = range(len(void_acc))

plt.plot(epoch_void,void_acc,'b',label='Training')
plt.plot(epoch_void,val_void_acc,'r',label='Validation')
plt.title('MSE of Void')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


void_loss = history.history['void_loss']
val_void_loss = history.history['val_void_loss']
epoch_void = range(len(void_loss))

plt.plot(epoch_void,void_loss,'b',label='Training Loss')
plt.plot(epoch_void,val_void_loss,'r',label='Validation Loss')
plt.title('MSE of Void Ratio')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



def LabelOutput(O):
    O = np.squeeze(np.array(O.argmax(axis=1)))
    return O

#y_void_test = (y_void_test_norm/100)*(max(y_void_test)-min(y_void_test))+min(y_void_test)

### Prediction with test data ###

img_index = 120

y_density_test_label = y_density_test

print("Actual Density:", density_dict[y_density_test_label[img_index]], "Actual Void Ratio:", y_void_test[img_index])

# predict 1 image
#pred = model.predict(X_test[img_index].reshape(1,128,128,1))
pred = model.predict([X_test])
# pred[0]: 1st output
# pred_density = density_dict[round(pred[0][0][0])]
pred_densitys = LabelOutput(pred[0])
pred_void = pred[1].astype('float64')[:,0]

#pred_void = (pred_void/100)*(max(pred_void)-min(pred_void))+min(pred_void)


from sklearn import metrics

# void_test_loss, void_test_accuracy = model.evaluate(X_test,y_void_test)
# density_test_loss, density_test_accuracy = model.evaluate(X_test,y_density_test)

# print("The test accuracy is\n", void_test_accuracy)
# print("The density test accuracy is\n", density_test_accuracy)
acc_density = metrics.accuracy_score(y_density_test_label,pred_densitys)
acc_void = metrics.mean_squared_error(y_void_test, pred_void)

print("Density Accuracy: ", acc_density)
print("Void MSE: ", acc_void)

print("Predicted Density:", density_dict[pred_densitys[img_index]], "Predicted Void Ratio:", round(pred_void[img_index],3))
plt.axis('off')
plt.imshow(X_test[img_index].reshape(128,128))
plt.title(f'Actual Void Ratio: {str(round(y_void_test[img_index],3))}  Actual State: {density_dict[y_density_test_label[img_index]]}\n Predicted Void Ratio: {str(round(pred_void[img_index],3))}  Predicted State: {density_dict[pred_densitys[img_index]]}',fontsize = 10)

void_pe = PercentError(y_void_test, pred_void)
print("Percent Error:\n", np.mean(void_pe))

results = np.vstack((y_void_test,pred_void)).T

r_s = np.array([(y_void_test[i],pred_void[i]) for i in np.argsort(y_void_test)])

x_axis = np.array(np.arange(1,len(y_void_test)+1,1).astype(float))

plt.figure()
plt.scatter(x_axis,r_s[:,0],5,'r',label = "Actual Results")
plt.scatter(x_axis,r_s[:,1],5,'b',label = "Predicted Results")
plt.xlabel("Number of data")
plt.ylabel("Void Ratio(e)")
plt.legend()
plt.title("Porosity Results")
plt.show()

plt.plot(x_axis,results[:,0],'r',label = "Actual")  
plt.plot(x_axis,results[:,1],'b',label = "Predicted")  
plt.xlabel("Number of data")
plt.ylabel("Void Ratio(e)")
plt.legend()
plt.title("Porosity Results")
plt.show()

unique, counts = np.unique(pred_densitys, return_counts=True)

fig = plt.figure(figsize = (5, 5))
name_label = {'Dense':81, 'Medium':81,'Loose':83}

pred_ds = {'Dense':83, 'Medium':76,'Loose':86}

plt.bar(list(name_label.keys()), list(name_label.values()), color ='b', 
        width = 0.3)
plt.bar(list(name_label.keys()), list(pred_ds.values()), color ='g', 
        width = 0.3)
 
plt.xlabel("Density State")
plt.ylabel("No. of State")
plt.title("Density State CNN Predictions")
plt.legend('Actual Density State', 'Predicted Density State')
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
#print(Max_Values)
#print(np.argmax([predictions]))
#print(confusion_matrix(Max_Values, y_test))

class_names = ['0:Dense',
               '1:Medium',
               '2:Loose']
#cm = confusion_matrix(y_test, Max_Values, labels=labels)
cm1 = confusion_matrix(y_density_test, pred_densitys, labels=[0,1,2])
print('cm is\n', cm1)

fig, ax = plt.subplots(figsize=(15,15)) 
#ax= plt.subplot()
#sns.set(font_scale=3)
#sns.set (rc = {'figure.figsize':(40, 40)})
sns.heatmap(cm1, annot=True, fmt='g', ax=ax, annot_kws={'size': 36})
ax.set_xlabel('True labels',fontsize = 25) 
ax.set_ylabel('Predicted labels',fontsize = 25)
ax.set_title('Confusion Matrix: CNN',fontsize = 36) 
ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 25)
ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 25)

# barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8)) 
 
# # set height of bar 
# test = [38,41,44] 
# pred_ds_2 = [122, 1, 0] 

 
# # Set position of bar on X axis 
# br1 = np.arange(len(test)) 
# br2 = [x + barWidth for x in br1] 

 
# # Make the plot
# plt.bar(br1, test, color ='g', width = barWidth, 
#         edgecolor ='grey', label ='Test Density State Label') 
# plt.bar(br2, pred_ds_2, color ='b', width = barWidth, 
#         edgecolor ='grey', label ='Predicted Density State Label') 

# plt.xlabel('Density State', fontweight ='bold', fontsize = 15) 
# plt.ylabel('No. of Density State', fontweight ='bold', fontsize = 15) 
# plt.xticks([r + barWidth for r in range(len(test))], 
#         ['Dense', 'Medium', 'Loose',])
# plt.legend()
# plt.show() 