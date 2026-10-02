import numpy as np


#===========================================================================
#basic layer structure 
#===========================================================================


def loss_attr(class_name: str, attr_name: str):
    
    return getattr(class_name, attr_name)

#===========================================================================

class layer:        #initialization
    def __init__(self, order, n_neurons, n_inputs, act_class, loss_class, inputs = []):  #don't want self inputs as an attribute as it overcomplicates initialization of the class. inputs determined outside of the initialization. 
        self.act = act_class
        self.loss = loss_class
        self.weights = np.random.randn(n_neurons, n_inputs)      
        self.biases = np.zeros(shape = (n_neurons, ))
        self.order = order      #what number/position layer the layer at hand is
        self.inputs = inputs
        self.nneur = n_neurons
        self.output = None

    def forward_propagate(self):     
        matmtpc = self.inputs@self.weights.T + self.biases
        return matmtpc

class output_layer(layer):

    def delta(self):
        dcda = self.loss.dc_da()
        ddz = self.act.d_dz()
        #print(f'shape ', np.shape(dcda), np.shape(ddz), np.shape(dcda@ddz))
        return dcda*ddz

    def dc_db(self, layers_list, activation_list):    #cost partial derivative with respect to bias
        delta = self.delta()
        self.dcdb = np.sum(delta, axis = 1)
        return self.dcdb

    def dc_dw(self, layers_list, activation_list):
        delta = self.delta()
        self.dcdw = delta.T @ self.inputs
        return self.dcdw


class hidden_input_layer(layer):

    def dc_db(self, layers_list, activation_list):# check here
        delta = self.act.delta_hidden(layers_list, activation_list)
        self.dcdb = np.sum(delta, axis = 1)
        return self.dcdb

    def dc_dw(self, layers_list, activation_list):
        delta = self.act.delta_hidden(layers_list, activation_list)
        self.dcdw = delta.T @ self.inputs

        return self.dcdw


#===========================================================================
#activation functions
#===========================================================================

class ReLU:
    def __init__(self, order, inputs = []):
        self.inputs = inputs
        self.order = order

    def forward(self):
        self.output = np.maximum(0,self.inputs)
        return self.output 

    def d_dz(self):
        ddz = np.where((self.inputs <= 0), 0.0, 1.0)
        
        return ddz


    def delta_hidden(self, layers_list, activations_list):

        output_index = (len(activations_list)-1)
        
        if self.order + 1 == output_index:
            #print(f'if statement = ', np.shape(layers_list[self.order+1].weights.T), np.shape(layers_list[self.order+1].delta()), np.shape(self.d_dz()))
            deltahidden = (layers_list[self.order+1].delta() @ layers_list[self.order+1].weights ) * self.d_dz()

        elif self.order < output_index:
            #print(f'elif statement = ', np.shape(layers_list[self.order+1].weights.T), np.shape(activations_list[self.order+1].delta_hidden(layers_list, activations_list)), np.shape(self.d_dz), np.shape(layers_list[self.order+1].weights.T @ activations_list[self.order+1].delta_hidden(layers_list, activations_list)))
            delta_next_hidden = (activations_list[self.order+1].delta_hidden(layers_list, activations_list) @ layers_list[self.order+1].weights )
            deltahidden = delta_next_hidden * self.d_dz()
        else:
            print(f'ERROR in calculating delta_hidden for activation function of order ', self.order)

        return deltahidden

        


class Sigmoid:
    def __init__(self, loss, order, input = []):
        self.lossclass = loss
        self.inputs = input
        self.order = order

    def forward(self):
        self.output = 1/(1+np.exp(-self.inputs))
        return self.output

    def d_dz(self):
        self.ddz = np.exp(-self.inputs)/((1+np.exp(-self.inputs))**2)
        return self.ddz

    def delta_output(self):
        deltasig = self.lossclass.dc_da()*self.d_dz()
        self.deltasigmoid = deltasig
        return self.deltasigmoid

    def delta_hidden(self, layers_list, activations_list):  #to be added
        pass
    

class Softmax:
    def __init__(self, loss, inputs):
        self.lossclass = loss
        self.inputs = inputs

    def forward(self, input):
        numerator = np.exp(input, keepdims = True) #===> returns e^z_i for each value, changes matrix into these values. 
        denominator = np.sum(input, axis = 1, keepdims = True) #===> returns the sum over all values in a batch of samples, puts it into a matrix 
        self.output = numerator/denominator         #keepdims helps numpy broadcast properly. good to remember. 
        return self.output

    def d_dz():     #to be added
        pass  

    def delta_output(self):
        deltasoft = self.lossclass.dc_da()*Softmax.d_dz(self.inputs)
        self.deltasoftmax = deltasoft
        return self.deltasoftmax

    def delta_hidden(self, layers_list, activations_list):      #To be added.
        pass


#===========================================================================
#loss functions
#===========================================================================


class Loss:         #this parent class is trivial right now, but the point is: 
                   #if I update this someday, I will now not have to implement a new parent loss class.
    pass

class MeanSquaredError(Loss):
    def __init__(self, layer_number, inputs = [0], targets = [0]):
        self.inputs = np.array(inputs)
        self.targets = targets
        self.layer = layer_number

    def error(self):
        squared_summed = np.sum((self.targets-self.inputs)**2)
        self.output = np.mean(squared_summed)
        return self.output

    def dc_da(self):   
        return 2*(self.inputs - self.targets) / self.inputs.size



class MeanAbsError(Loss):       #To be added
    def __init__(self, targets, inputs = [0]):
        self.inputs = inputs
        self.targets = targets

    def error(self):
        abs_summed = np.sum(np.abs(self.inputs - self.targets))
        self.output = np.mean(abs_summed)

    def delta():
        pass







     
def forward_prop(layers, activations):

    
    
    outputs_list = [activations[0].forward()]    
   
    for j in range(len(layers)-1):   
        layers[j+1].inputs = outputs_list[j]
        output_of_layer = layers[j+1].forward_propagate()
        activations[j+1].inputs = output_of_layer
        output_activated = activations[j+1].forward()
        outputs_list.append(output_activated)


    return outputs_list

#===========================================================================
#Implementation of Gradient Descent Algorithm
#===========================================================================

class optimizer:        
    def __init__(self): 
        pass


class GradientDescent(optimizer):           

    def optimize(self, epochs, learning_rate, layers_opt, activation_opt):
       
        
        for i in range(epochs):

            forward_prop(layers = layers_opt, activations = activation_opt)

            for j, layer in zip(range(len(layers_opt)), layers_opt):       

                w_new = layer.weights - learning_rate * layer.dc_dw(layers_list = layers_opt, activation_list = activation_opt)
                b_new = layer.biases - learning_rate * layer.dc_db(layers_list = layers_opt, activation_list = activation_opt)

                w = w_new
                b = b_new

                layer.weights = w
                layer.biases = b

        print(f'Final weights = ', w, 'Final biases = ', b)

