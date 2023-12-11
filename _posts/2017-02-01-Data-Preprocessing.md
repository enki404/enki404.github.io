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

  * ### **Rose Diagram and 2D Direction Distribution Histogram of Fabric Data: Cartesian Coordinate vs. Spehrical Coordinate**
 
Rose plots are the most popular method for plotting directional distribution data. These plots are essentially frequency histograms plotted in polar coordinates, using **`directional information`** from the total contact normal **`(nx, ny, nz)`** in the DEM simulation to illustrate particle interactions.

To facilitate connection with the CNN/CNN-GRU model, we will explore another way of presenting fabric data, namely mapping the 3D contact normal direction distribution to a 2D image. The direction value of each contact normal **`(nx, ny, nz)`** in Cartesian coordinates is converted to spherical coordinates **`(θ, φ)`**. The workflow of coordinate conversion is shown in the figure. The direction vector of the total contact normal can be converted using the following equation:

$$ \theta = atan(\frac{\sqrt[2]{nx^2+ny^2}}{nz}) $$

$$ \phi = atan2(ny,nx) $$


* 
