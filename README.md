# Neural-Decision-Forests
An implementation of the Deep Neural Decision Forests(dNDF) in PyTorch.
![](http://cnyah.com/2018/01/29/dNDF/arch.png)

# Features
- Two stage optimization as in the original paper [Deep Neural Decision Forests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf) (fix the neural network and optimize $\pi$ and then optimize $\Theta$ with the class probability distribution in each leaf node fixed )
- Jointly training $\pi$ and $\Theta$ proposed by chrischoy in his work [Fully Differentiable Deep Neural Decision Forest](https://github.com/chrischoy/fully-differentiable-deep-ndf-tf)
- Shallow Neural Decision Forest (sNDF)
- Deep Neural Decision Forest (dNDF)

# Datasets
MNIST, UCI_Adult, UCI_Letter and UCI_Yeast datasets are available. For datasets other than MNIST, you need to go to corresponding directory and run the `get_data.sh` script.

# Requirements
- Python 3.x
- PyTorch 0.3.x
- numpy
- sklearn


# Usage
 ```
 python train.py --ARG=VALUE
 ```

 in the case of training the sNDF on MNIST with alternating optimization, the command is like
 
 ```
 python train.py -dataset mnist -n_class 10 -gpuid 0 -n_tree 80 -tree_depth 10 -batch_size 1000 -epochs 100
 ```

# Results

Not spending much time on picking hyperparameters and without bells and whistles, I got the accuracy results(obtained by training $\pi$ and $\Theta$ seperately) as follows:

| Dataset | sNDF | dNDF | 
| - | :-: | -: | 
| MNIST | 0.9794| 0.9963 | 
| UCI_Adult | 0.8558 | NA | 
| UCI_Letter | 0.9507 | NA |
| UCI_Yeast | 0.6031 | NA |

By adding the nonlinearity in the routing function, the accuraries can reach 0.6502 and 0.9753 respectively on the UCI_Yeast and UCI_Letter.

# Note
Some people may experience the 'loss is NaN' situation which could be caused by the output probability being zero. Please make sure you have normalized your data and used a large enough tree size and depth. In the case that you want to stick with your tree setting, a workaround could be to clamp the output value. 
