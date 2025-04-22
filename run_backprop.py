import numpy as np
import matplotlib.pyplot as plt
from utils import *
from feedforward import *
import pickle as pkl

def train(low_res=False, alpha=1e-2, num_epochs=100, batch_size=32):

    X, t_raw = load_X_t(filepath='digits10000.mat', lower_res=low_res)
    t = np.array([hot_one_encode(value) for value in t_raw])

    if low_res:
        input, hidden1, hidden2, output = 196, 64, 32, 10
    else:
        input, hidden1, hidden2, output = 784, 200, 80, 10

    net = Network(input=input, hidden1=hidden1, hidden2=hidden2, output=output, alpha=alpha, batch_size=batch_size)

    # fig, ax, line = setup_live_plot()
    # line, = ax.plot([], [], label='Cost')
    # rolling_line, = ax.plot([], [], 'k--', label='Rolling Avg')
    # 
    costs = []
    rolling_sum = 0
    window_size = 50

    for epoch in range(num_epochs):

        X, t_raw = shuffle_data(X, t_raw)
        t = np.array([hot_one_encode(value) for value in t_raw])
        total_epoch_cost = 0

        for i in range(X.shape[0]):

            #clear gradients and preds
            net.zero_gradients()

            #run forward pass through batch
            y_pred = net.forward(X[i, :])
            cost = net.backward(t[i,:], X[i, :])
            net.update(1)  # Update weights and biases after processing the batch


            total_epoch_cost += cost
            costs.append(cost)

            rolling_sum += cost
            if len(costs) > window_size:
                rolling_sum -= costs[- window_size -1]
            rolling_avg = rolling_sum / window_size if len(costs) > window_size else None
            
            max_points = 1000

            if i % 100 == 0:
                print(f"Iteration {i}:")
                print(f"Predicted: {y_pred}")
                print(f"True: {t[i,:]}")
                print(f"Loss: {cost}\n")

            #     # Trim costs list to a max length
            #     if len(costs) > max_points:
            #         costs = costs[-max_points:]

            #     # Update loss line
            #     line.set_data(range(len(costs)), costs)

            #     # Update rolling average
            #     if len(costs) >= window_size:
            #         rolling_avg = np.convolve(costs, np.ones(window_size)/window_size, mode='valid')
            #         rolling_line.set_data(range(window_size - 1, len(costs)), rolling_avg)
            #     else:
            #         rolling_line.set_data([], [])

            #     ax.relim()
            #     ax.autoscale_view()
            #     plt.pause(0.01)

        if total_epoch_cost < 0:  # If loss goes negative, something is wrong
            print(f"Warning: Total cost went negative at epoch {epoch}")
            break

        pkl.dump(net, open(f'nets/net{epoch}.pkl', 'wb'))
        net.zero_gradients()

    plt.ioff()  # Turn off interactive mode
    plt.show()

def run(filepath):
    # Load the model
    net = pkl.load(open(filepath, 'rb'))

    # Load the data
    X, t_raw = load_X_t(filepath='digits10000.mat', lower_res=True)
    t = np.array([hot_one_encode(value) for value in t_raw])

    X = X[:1000, :]
    t = t[:1000, :]

    losses=[]
    # Test the model
    for i in range(X.shape[0]):
        y_pred = net.forward(X[i, :])
        loss = net.backward(t[i,:], X[i, :])
        losses.append(loss)

    losses = np.array(losses)
    print(np.mean(losses))

if __name__ == "__main__":
    run('nets/net99.pkl')
    #train(low_res=True, alpha=1e-2, batch_size=1)



