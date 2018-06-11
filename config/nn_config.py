import os

def get_nn_params():
    nn_params = {}
    drummer = "jack-dejohnette"
    nn_params['model type'] = 'lstm'
    nn_params["drummer"] = drummer

    nn_params["epochs"] = 3
    nn_params['batchsize'] = 10
    nn_params["sequence length"] = 100

    # 'audio'
    nn_params["sourcedir"] = "audio/" + drummer + "/"
    nn_params["slicedir"] = "audio/" + "sliced/" + drummer + "/"

    # 'midi'
    nn_params["midi dir"] = "midi/" + drummer + "/"

    # "np-data/'
    nn_params["activationsdir"] = "np-data/activations/" + drummer + "/"
    nn_params["onsetsdir"] = "np-data/onsets/" + drummer + "/"
    nn_params["training data dir"] = 'np-data/training-data/' + drummer + "/"

    nn_params["dev data dir"] = "np-data/dev-data/" + drummer + "/"

    # 'plots/'
    nn_params["ADT sequence directory"] = "plots/ADT/" + drummer + "/"
    nn_params["MIDI sequence directory"] = "plots/MIDI/" + drummer + "/"
    nn_params["loss directory"] = "plots/loss/" + drummer + "/"
    # network architecture

    #nn_params["nn_architecture"] = "lstm"
    #nn_params["epochs"] = 50 # number epochs to train

    # 'model-weights'
    nn_params["weightsdir"] = 'model-weights/' + drummer + "/"

    # 'midi'
    nn_params["rendered midi dir"] = 'midi/rendered-' + drummer + "/"

    nn_params["models dir"] = "models/" + drummer + "/"

    model_types = ['lstm', 'stacked-lstm', 'stateful-lstm-time-distributed', 'stateful-lstm', 'cwrnn']

    for directory in nn_params:
        #print nn_params[directory]
        if directory not in model_types:
            if os.path.exists(str(nn_params[directory])) == False and isinstance(nn_params[directory], basestring):
                print "creating %r directory" %(nn_params[directory])
                os.makedirs(nn_params[directory])

    for elem in nn_params:
    	print "%r : %r" %(elem, nn_params[elem])
    return nn_params
#
# def get_hyperparams(){
#     nn_params = get_nn_params()
#     drummer = nn_params["drummer"]
#     hp_params = {}
#     hp_params["model type"] = "stacked_model"
#
# }

#nn_params = get_nn_params()
