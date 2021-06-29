# MLP-AI-Network
 MLP network with two hidden layers

Multilayer Perceptron network.
With two hidden layers.
The first one has as activated function the tanh(u),
and the second layer has the log or linear function (you can change it before the training). 

Given points in 2D space, divided in four groups with same random error (default 10%)
All four points sets are generated by the program automatically.

To run the program you must do make all.
Then run the neural network by <./nn>
Press <help> to see what you can do,
    but teh main commands are:
    <define> or <define default> : todefine how many neurons in each layer you want to have.
    <train> : to train your MLP network
    <save> : if you are satisfied with the result, and you want to save neurons's waights
    <test> : to test your MLP network's efficiency

Same of results of the network for (H1,H2)->{(7,4),(8,5),(10,6)}
    and for different size of beaches (1,N/10,N/100,N):

      |B |         1       |        N/10     |        N/100    |         N       |
H1,H2 |  | logistic|linear | logistic|linear | logistic|linear | logistic|linear | 
 7,4  |  |  96.6%  | 91.8% |  95.9%  | 88.2% |  93.5%  | 93.0% |  96.1%  | 91.1% |
 8,5  |  |  96.4%  | 95.8% |  96.3%  | 88.5% |  97.1%  | 93.5% |  97.0%  | 88.3% |
10,6  |  |  97.2%  | 96.6% |  97.2%  | 92.1% |  97.0%  | 96.5% |  97.0%  | 96.4% |