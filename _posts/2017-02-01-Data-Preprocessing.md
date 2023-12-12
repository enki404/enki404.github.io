---
title:  "Data-Preprocessing"
mathjax: true
layout: post
categories: media
order: 2
---

---

# Data Preprocessing

* ## **Data Gathering**

  * ### **Rose Diagram and 2D Contact Direction Distribution Histogram of Fabric Data: Cartesian Coordinate vs. Spehrical Coordinate**
 
  Rose plots are the most popular method for plotting directional distribution data. These plots are essentially frequency histograms plotted in polar coordinates, using **`directional information`** from the total contact normal **`(nx, ny, nz)`** in the DEM simulation to illustrate particle interactions.
 
  To facilitate connection with the CNN/CNN-GRU model, we will explore another way of presenting fabric data, namely mapping the 3D contact normal direction distribution to a 2D image. The direction value of each contact normal **`(nx, ny, nz)`** in Cartesian coordinates is converted to spherical coordinates **`(θ, φ)`**. The workflow of coordinate conversion is shown in the [Figure 1]. The direction vector of the total contact normal can be converted using the following equation:
 
  **$$ \theta = atan(\frac{\sqrt[2]{nx^2+ny^2}}{nz}) $$**
 
  **$$ \phi = atan2(ny,nx) $$**
 
  [Figure 2] displays the representative rose diagram in 3D space based on cartesian coordinate. 
 
  The rose diagram can be directly mapped to a frequency histogram in 3D space using converted angles (θ, ϕ) based on spherical coordinate. We can then convert these 3D histograms to 2D images by using color scale to indicate the height of each bin in the histogram. The horizontal axis represents θ within the range of [-90°, 90°], while the vertical axis corresponds to ϕ within the range of [-180°, 180°]. [Figure 3] illustrates the representation of 3D directional distribution histograms and the corresponding 2D histogram images.


  * ### **Raw 2D Contact Direction Distribution Histogram Image Dataset**
    
  Open-source software YADE (Šmilauer et al, 2015) is used for all DEM simulations in this study. An assembly of 10,000 spherical particles is generated in a cubical domain using the prescribed grain size distribution (GSD)。Periodic boundaries are used in the simulation to ensure that particle information is not lost due to boundary effects. Initially after generation, all particles are not in contact and are thus in a free state. Then an isotropic (or K0) consolidation procedure is performed until a target mean stress (p) value is achieved through the servo-controller set by YADE. Samples with different initial void ratios (e0) can be obtained by changing the coefficient of friction μ between particles during consolidation, and the typical μ value for quartz sand 0.5 is restored right before the shearing stage.

  In this study, we prepared dense, medium, and loose samples using 

  $$ μ_{0} = 0.0 (Dense), 0.2(Medium), 0.5(Loose) $$ 

  The shear stage is conducted following the conventional triaxial compression condition with `intermediate principal stress ratio b=0`. 

  The magnitude of cyclic loading is determined by `cyclic stress ratio` , which is calculated by (deviatoric stress) and (principal stress)

  $$ CSR = \frac{q}{2p_{0}} $$ 

  $$ q = σ_{1}-σ_{3} $$ 

  In this study, a CSR = 0.2 is adopted for all tests.The Cyclic Drained Triaxial Tests (Cyclic-DTX) are performed under a set of different confining pressures 

  $$ p_{0} = 50, 100, 300, 500, 700, 1000, 1500, 2000 (kPa) $$

  All tests are conducted to a maximum number of cycles (N=100). The information of stress, strain, void ratio, and fabric tensor are monitored throughout the loading program. The maximum deviatoric stress during cyclic loading is 

  $$ q_{max}=(2p_{0})*CSR = 0.4p_{0} $$

  After the cyclic loading stage, all specimens are sheared to a large value of axial strain 

  $$ ε_{a} = 50{\\%} $$

  to achieve the apparent critical state condition by monotonic loading.

  **A total of 24 tests were performed and we selected void fraction information under monotonic loading, in which case the void fraction was reported as a step increase from 0% to 50% with axial strain. Therefore, we can obtain 51 two-dimensional fabric orientation distribution histogram images for each simulation, and a total of **`51x24 = 1224`** original images are obtained through this method.**

*  ## **Models/Methods**

The network progresses by moving inputs forward, activating neurons positively, ultimately producing the final output. This forward movement is known as the forward pass or forward propagation. Following this, errors are computed by comparing actual data with model outputs. These errors are then propagated backward through the network, a process referred to as backpropagation. During backpropagation, to optimize the loss and achieve minimal loss, weights must be updated proportionate to their contribution to the error. This update process employs the gradient, which represents the derivative of the error with respect to the weights. A single iteration of the above process is termed an epoch. The number of epochs is determined based on the specific objectives of the work.

Typically, errors can be accumulated across training examples and updated collectively at the end of a batch, a process referred to as batch learning. The batch size corresponds to the computational quantity of examples involved in each update. The extent of weight adjustment is regulated by the learning rate, often referred to as a configuration parameter or step size. The designed model are employed as outlined in Figure 4, based on the preceding information.

| Layer (type)                   |Output Shape             | Param #     | Connected to                      |
| ------------------------------ | ----------------------- |-------------|-----------------------------------|
| Input (InputLayer)             |[(None, 128, 128, 1)     |      0      |      []                           |              
| conv2d (Conv2D)                |(None, 128, 128, 32)     |     832     |  ['Input[0][0]']                  |
| batch_normalization            |(None, 128, 128, 32)     |     128      |  ['conv2d[0][0]']                |
| max_pooling2d (MaxPooling2D)   |(None, 42, 42, 32)       |      0      |  ['batch_normalization[0][0]']    |  
| conv2d_1 (Conv2D)              |(None, 42, 42, 64)       |    51264    |  ['max_pooling2d[0][0]']          |  
| batch_normalization_1          |(None, 42, 42, 64)       |     256     |  ['conv2d_1[0][0]']               | 
| max_pooling2d_1 (MaxPooling2D) |(None, 14, 14, 64)       |      0      |  ['batch_normalization_1[0][0]']  |             
| conv2d_2 (Conv2D)              |(None, 14, 14, 128)      |    73856    |  ['max_pooling2d_1[0][0]']        |  
| max_pooling2d_2 (MaxPooling2D) |(None, 7, 7, 128)        |      0      |  ['conv2d_2[0][0]']               |  
| conv2d_3 (Conv2D)              |(None, 7, 7, 256)        |    295168   |  ['max_pooling2d_2[0][0]']        |  
| max_pooling2d_3 (MaxPooling2D) |(None, 3, 3, 256)        |      0      |  ['conv2d_3[0][0]']               |  
| flatten (Flatten)              |(None, 2304)             |      0      |  ['max_pooling2d_3[0][0]']        |
| dense (Dense)                  |(None, 64)               |    147520   |  ['flatten[0][0]']                |  
| batch_normalization_4          |(None, 64)               |     256     |  ['dense[0][0]']                  |         
| dropout (Dropout)              |(None, 64)               |      0      |  ['batch_normalization_4[0][0]']  |   
| dense_1 (Dense)                |(None, 64)               |     4160    |  ['dense[0][0]']                  |  
| dropout_1 (Dropout)            |(None, 64)               |      0      |  ['dropout[0][0]']                |  
| dropout_2 (Dropout)            |(None, 64)               |      0      |  ['dense_1[0][0]']                |  
| density (Dense)                |(None, 3)                |     195     |  ['dropout_1[0][0]']              |  
| void (Dense)                   |(None, 1)                |      65     |  ['dropout_2[0][0]']              |  
|Total Params: 573,700           |Trainable params: 573,380| Non_Trainable params: 320                       |                                                                                                 

The structural details of the CNN model in TF/Keras are shown in the code section below. It is worth noting that we try to use the same model structure to implement both the classification problem and the regression problem by only changing the output layer, partial fully connected layer and dropout layer. Therefore, this CNN model does not use the traditional `model.sequential()` to connect the neural network structure. In a sense, the model built here is more similar to the `(sequential + parallel)` structure.


  ````
  ```javascript
  inputs = Input((input_shape),name='Input')
  
  `Convolutional Layers`
  
  conv1 = Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same')(inputs)
  batch1 = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.009)(conv1)
  maxp1 = MaxPooling2D(pool_size=(3,3))(batch1)
  
  conv2 = Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same')(maxp1)
  batch2 = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.15)(conv2)
  maxp2 = MaxPooling2D(pool_size=(3,3))(batch2)
  
  conv3 = Conv2D(128,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(maxp2)
  bacth3 = tf.keras.layers.BatchNormalization()(conv3)
  maxp3 = MaxPooling2D(pool_size=(2,2))(conv3)
  
  conv4 = Conv2D(256,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(maxp3)
  batch4 = tf.keras.layers.BatchNormalization()(conv4)
  maxp4 = MaxPooling2D(pool_size=(2,2))(conv4)
  
  `Flatten Layer `
  flatten = Flatten()(maxp4)
  
  `Fully Connected layer`
  
  fc1 = Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.09))(flatten)
  batch5 = tf.keras.layers.BatchNormalization()(fc1)
  dropout1 = Dropout(0.4)(batch5)
  fc2 = Dense(64,activation='relu')(flatten)
  
  dropout11 = Dropout(0.7)(dropout1)
  dropout2 = Dropout(0.05)(fc2)
  
  `Regression $ Classificcation Outputs`
  
  output1 = Dense(3,activation='softmax',name='density')(dropout11)
  output2 = Dense(1,activation='linear',name='void')(dropout2)
  
  model = Model(inputs = [inputs], outputs = [output1,output2])
  }
  ```
  ````



