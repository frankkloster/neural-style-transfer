This code base is largely an implementation of the neural style algorithm. This is largely based off the following problem, given some image, how can I redraw that image in another style? Here, we go through the process of giving two images, one content image (which is the object you wish to draw), and one style image (which contains a style you wish to emulate). What you get is a third image, with the object in the content image redrawn in the same style as the style image. This is best diagramatically described below.

![](https://pytorch.org/tutorials/_images/neuralstyle.png)

# Instructions
The algorithm is largely ran off the main file. Several options are avaliable.

Option          | Abbreviation | Function
------          | -------- | -----
--content_path  | -c       | Path to the content file.
--style_path    | -s       | Path the the style file.
--final         | -f       | Path to the final saved image.
--alpha         | -a       | Content Weight
--beta          | -b       | Style Weight

For instance, 

```
$ python main.py -c content.jpg -s style.jpg -f combined.jpg
```


# References
- https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398 <br> Turorial explaining some of the material. <p>
- [A Neural Algorithm of Artistic Style - Leon A. Gatys, Alexander S. Ecker, Matthias Bethge](https://arxiv.org/abs/1508.06576) <br> The original paper describing neural style transfer. <p>
- [Tensorflow's Models Repository](https://github.com/tensorflow/models) <br> Very cool repository. Much of the code is based off the source code for their neural style transfer code base, found inside their research models. A bunch of other nice models include their object detection model.<p>
- Deep Learning with Python, Francois Chollet <br> This is the first book that introduced me to the idea of neural style transfer. Here, it is coded in Scipy, and a BFGS algorithm to optimize the loss function.