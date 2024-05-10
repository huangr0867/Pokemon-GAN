# **Pokemon GAN**

A DCGAN model was implemented here using Tensorflow 2.0. Pokemons are generated using a dataset of 7536 pokemons. Some sample pokemons generated:

<a href="url"><img src="dcgan/results/epoch_049.png" align="center" height="500" width="500" ></a>

### **Prerequisites**

1. Enter the dcgan directory

```
cd dcgan
```

2. Change the config.py file to fix your specific path files and configurations like the number of epochs
3. Unzip the data_ready.zip file in the data dir. This contains a subset of the actual dataset (503 images). Please contact ryan_y_huang[at]brown.edu for the entire set of images if you want to replicate our results

### **Train**

1. Again, enter the dcgan directiory

2. Run

```
python main.py
```
