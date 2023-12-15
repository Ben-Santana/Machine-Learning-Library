from layer_def import Dense, Tanh, mse, mse_prime
import numpy as np
from network import train, predict



#create dataset
#reshape to make the data 2, 4
X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


#create network
network = [
    Dense(2, 3), 
    Tanh(), 
    Dense(3, 1), 
    Tanh()
]

#train
train(network, mse, mse_prime, X, Y, epochs=50000, learning_rate=0.1)

#--------------------------------------------------------------------------------
# 🍝 code for testing xor in terminal

print("To exit trials at any point type 2")

while True:
    input_one = int(input("First Value: "))
    if (input_one == 2): break
    input_two = int(input("Second Value: "))
    if (input_two == 2): break

    trial = [input_one, input_two]
    result = 0;
    for i in predict(network, trial):
        for b in i:
            result += b
    if( result < 0.5 ): print(0)
    else: print(1)
