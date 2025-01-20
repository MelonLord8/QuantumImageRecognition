import pennylane as qml
from pennylane import numpy as qnp
import jax
from jax import numpy as jnp
import optax
import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

num_wires = 6
num_rot = 3
num_layers = 12

dev = qml.device("default.qubit", wires = num_wires)

def get_data(dataset):
    df = pd.read_csv(dataset, dtype = int, header= None, index_col = None)
    y_train = qnp.array(df.iloc[:,0])
    x = df.iloc[:,1:]
    x_train = qnp.array([qnp.array(x.iloc[:,2*i]) + qnp.array(1j*x.iloc[:,2*i + 1]) for i in range(x.shape[1]//2)]).T
    return x_train, y_train

def embedder(x, padding):
    qml.AmplitudeEmbedding(x, wires=[0,1,2,3,4,5], normalize = True, pad_with = padding)

def ansatz(params):
    qml.StronglyEntanglingLayers(params, wires=[0,1,2,3,4,5])

observable = qnp.asarray([[0,0],
                        [0,1]])

@qml.qnode(dev, interface = "jax")
def quantum_network(params, x):
    params_conv = params["params_conv"] 
    params_dev = params["params_dev"]
    conv_matrix = jnp.stack([jnp.concatenate([jnp.zeros(shape=(56*(i//7),),dtype=jnp.complex64),
                                              jnp.concatenate([jnp.concatenate([
                                                                    jnp.zeros(shape = (2*(i%7),)),
                                                                    params_conv[2*j : 2*(j+1)],
                                                                    jnp.zeros(shape = (12 - 2*(i%7), ))]) for j in range(4)]), 
                                              jnp.zeros(shape=(392 - 56*(i//7+1),) , dtype=jnp.complex64)]) for i in range(49)])
    conv_x = jnp.matmul(conv_matrix,x.T)
    embedder(conv_x.T, params["padding"][0])
    ansatz(params_dev)
    return qml.expval(qml.Hermitian(observable, wires=[0]))

@jax.jit
def mse(params, x, y):
    y_pred = quantum_network(params, x)
    return (y - y_pred)**2

mse_map = jax.vmap(mse, in_axes = (None, 0, 0))

@jax.jit
def loss_fn(params, x, y):
    return jnp.mean(mse_map(params,x,y))

params_dev_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers,n_wires=num_wires)

learning_rate_schedule = optax.schedules.join_schedules(schedules = [optax.constant_schedule((1/10)**i) for i in range(6)],
                                                                     boundaries = [10, 100, 250, 450, 700])

opt = optax.adam(learning_rate = learning_rate_schedule)
max_steps = 1000

@jax.jit
def optimiser(params, data, training , print_training):
    opt_state = opt.init(params)
    args = (params, opt_state, jnp.asarray(data), jnp.asarray(training),print_training)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, max_steps+1, update_step_jit, args) 
    return params

@jax.jit
def update_step_jit(i,args):
    # Unpacks the arguments
    params, opt_state, data, targets, print_training = args
    # Gets the loss and the gradients to be applied to the parameters, by passing in the loss function and the parameters, to see how the parameters perform 
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    #Prints the loss every 25 steps if print_training is enable
    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=jnp.sqrt(loss_val))
    jax.lax.cond((jnp.mod(i, 50) == 0 ) & print_training, print_fn, lambda: None)
    #Applies the param updates and updates the optimiser states
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    #Returns the arguments to be resupplied in the next iteration
    return (params, opt_state, data, targets, print_training)   

init_params = {
    "params_conv": qnp.random.default_rng().random(size = (8,)),
    "params_dev": qnp.random.default_rng().random(size = params_dev_shape),
    "padding": qnp.random.default_rng().random(size = (1,))*32
}

x_train, y_train = get_data("data/train.csv")
print("Starting training")
opt_params = optimiser(init_params, x_train, y_train, True)
x_test, y_test = get_data("data/test.csv")
print(loss_fn(opt_params, x_test, y_test))
with open("opt_params.pkl", "wb") as f:
    pkl.dump(opt_params, f)