import numpy as np
import logging.config
import torch
import time

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def adjust_optimizer(optimizer, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):

        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    config['lr'] = config['lr'] * config['lr_decay']
    optimizer = modify_optimizer(optimizer, config)

    return optimizer

def save_parameters(model, save_path=None):
    # Save the parameters in a similarly npz file build from Theano and Lasagne
    array=[]
    param=model.state_dict()
    for i in range(6):
        conv_weights=np.array(param['features.conv_%i.weight' %i])
        array.append(conv_weights)
        if model.features['conv_%i' % i].bias is not None:
            conv_bias = np.array(param['features.conv_%i.bias' %i])
        else:
	    #bias(k)=0
            conv_bias=np.zeros(model.features['conv_%i' % i].out_channels, dtype=np.float32)
        if model.features['batchnorm_%i' %i].affine:
            batchnorm_bias = np.array(param['features.batchnorm_%i.bias' %i])
            batchnorm_weights = np.array(param['features.batchnorm_%i.weight' %i])
        else:
	    # β(k) = 0 and γ(k) = 1
            batchnorm_bias = np.zeros(model.features['conv_%i' % i].out_channels, dtype=np.float32)
            batchnorm_weights = np.ones(model.features['conv_%i' % i].out_channels,dtype=np.float32)
        batchnorm_running_mean=np.array(param['features.batchnorm_%i.running_mean' %i])
        batchnorm_running_var = np.array(param['features.batchnorm_%i.running_var' % i])
        array.append(conv_bias)
        array.append(batchnorm_bias)
        array.append(batchnorm_weights)
        array.append(batchnorm_running_mean)
        array.append(np.sqrt(batchnorm_running_var)**-1)

    for i in range(3):
        fc_weights=np.array(param['classifier.fc_%i.weight' %i])
        array.append(fc_weights.transpose())
        if model.classifier['fc_%i' % i].bias is not None:
            fc_bias = np.array(param['classifier.fc_%i.bias' % i])
            array.append(fc_bias)
        else:
            #no bias for the last layer
            if i<2:
		# bias(k)=0
                fc_bias = np.zeros(model.classifier['fc_%i' % i].out_features, dtype=np.float32)
                array.append(fc_bias)
        if model.classifier['batchnorm_%i' % i].affine:
            batchnorm_bias = np.array(param['classifier.batchnorm_%i.bias' % i])
            batchnorm_weights = np.array(param['classifier.batchnorm_%i.weight' % i])
        else:
	    # β(k) = 0 and γ(k) = 1
            batchnorm_bias = np.zeros(model.classifier['fc_%i' % i].out_features,dtype=np.float32)
            batchnorm_weights = np.ones(model.classifier['fc_%i' % i].out_features, dtype=np.float32)
        batchnorm_running_mean=np.array(param['classifier.batchnorm_%i.running_mean' %i])
        batchnorm_running_var = np.array(param['classifier.batchnorm_%i.running_var' % i])
        array.append(batchnorm_bias)
        array.append(batchnorm_weights)
        array.append(batchnorm_running_mean)
        array.append(np.sqrt(batchnorm_running_var)**-1)

    np.savez(save_path,*array)

def quantized_training(model, optimizer, criterion, train_loader, test_loader,
		       num_epochs, device, save_path=None):
    best_epoch = 0
    best_test_err = 100
    accuracy = []
    train_loss=[]
    test_loss_epoch=[]
    for epoch in range(num_epochs):
        model.train()
        exe_time = time.time()
        train_loss_epoch = []
        for i, sample in enumerate(train_loader):
            images = sample['image']
            labels = sample['label']

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.clipWeights()
            train_loss_epoch.append(loss.item())

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sample in test_loader:
                images = sample['image']
                labels = sample['label']
                images = images.to(device)
                labels = labels.to(device)
                bnn, outputs = model.evaluation(images)
                predicted = torch.argmax(bnn.data, 1)
                loss = criterion(outputs, labels)
                test_loss_epoch.append(loss.item())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        #Display the results for the epoch
        epoch_duration = time.time() - exe_time
        err_epoch = 100 * (1 - correct / total)
        accuracy.append(correct / total)
        train_loss.append(np.mean(train_loss_epoch))
        test_loss=np.mean(test_loss_epoch)
        if err_epoch <= best_test_err:
            loss_best_epoch = test_loss
            best_epoch = epoch
            best_test_err = err_epoch
            if save_path is not None:
                save_parameters(model,save_path+'.npz')
                torch.save(model.state_dict(), save_path+'.ckpt')
        print("Epoch " + str(epoch + 1) + " of " + str(num_epochs) + " took " + str(epoch_duration) + "s")
        print("  LR:                            " + str(optimizer.param_groups[0]['lr']))
        print("  training loss:                 " + str(train_loss[-1]))
        print("  best epoch:                    " + str(best_epoch + 1))
        print("  best test error rate:          " + str(best_test_err) + "%")
        print("  best test loss:                " + str(loss_best_epoch))
        print("  test error rate:               " + str(err_epoch) + "%")
        print("  test loss:                     " + str(test_loss))

        optimizer = adjust_optimizer(optimizer, model.hyper_parameters)

    return (train_loss, accuracy)
