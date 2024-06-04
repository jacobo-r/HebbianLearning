import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy

import torchvision
import torchvision.transforms as transforms

import os
import time

import numpy as np
import time
from datetime import datetime, timedelta
import sys

# this version of the code was created to correct the dumb mistake of the correlation weight : instead of [-1,1] now we do this +/- 1
# in order to augment the desired updates (meaning if threshold =0 , then coeff belogns to [-2,-1] U [1,2])
# So here we are just doing that and we are also storing the correlation statistics to see how correlated inputs are :)
# Maybe ther are too correlated and hence we are just multiplying by the same number the whole thing = bad
# Maybe on the contrary nothing is happening :)
# The threshold I will use here is 0.6
#I will store the stats on the correlations.

def main():
      # Check if any arguments were passed (beyond the script name)
    if len(sys.argv) > 1:
        print(f"Arguments received: {sys.argv[1:]}")
        print(f"For Number of Layers:{sys.argv[1]}, threshold: {sys.argv[2]}")

        layers = sys.argv[1]
        correlation_threshold = sys.argv[2]
    else:
        print("No arguments received.")


    # DATASET INITALIZATION ///////////////////////////////////
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    BATCHSIZE=100

    transform_train = transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCHSIZE, shuffle=True, num_workers=2) # Also check out pin_memory if using GPU
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCHSIZE, shuffle=False, num_workers=2)


    # NETWORK INITALIZATION ///////////////////////////////////

    NL= int(layers)          # Number of layers 
    STRIDES=(1,1,1)         # Input strides (for each layer)    
    POOLSTRIDES = (2, 2, 2) # Pooling strides
    POOLDIAMS = (2, 2, 2)   # Pooling diameters
    SIZES = (5, 3, 3)       # Receptive field sizes
    N = [100, 196, 400]     # N = number of cells per column (i.e. channels) for each layer
    K = [1, 1, 1]           # K = number of winners in the winner-take-all (WTA) competition

    TARGETRATE = [float(K[ii] / N[ii]) for ii in range(NL)]   #  The target winning rate of each cell *must* be K/N to allow for equilibrium

    CSIZE = 32              # Input image size
    FSIZE = 12              # Size of the Difference-of-Gaussians filter
    NBLEARNEPOCHS = 5       # Number of epochs for Hebbian learning (2 more epochs are added for data acquisition for training and testing the linear classifier)
    LR = 0.01 / BATCHSIZE   # Learning rate
    MUTHRES =   10.0        # Threshold adaptation rate

    USETRIANGLE = False     # Should we use Coates' "triangle" method to compute actual neural responses, or should we just use the WTA result as is?
    
    total_batches = total_batches = len(trainset) // BATCHSIZE * NBLEARNEPOCHS -100

    tic = time.time()

    # Initializations
    w=[]
    optimizers=[]
    thres=[]
    for numl in range(NL):
        NIC = 1 if numl == 0 else N[numl-1]  # Number of input channels for the weight filters at each layer
        wi = torch.randn((N[numl], NIC, SIZES[numl], SIZES[numl]), requires_grad=True, device=device) 
        wi.data = wi.data  / (1e-10 + torch.sqrt(torch.sum(wi.data ** 2, dim=[1,2,3], keepdim=True))) # Weights have norm 1
        w.append(wi)
        thres.append(torch.zeros((1,N[numl],1,1), requires_grad=False).to(device))  # Thresholds (i.e. adaptive biases to ensure a roughly equal firing rate)
        optimizers.append(optim.SGD((w[numl],), lr=LR, momentum=0.0))       # We use one optimizer per layer, though this is not strictly necessary with judicious use of .detach()              


    # Build a difference-of-Gaussians kernel to spatially decorrelate (approximately whiten) the images
    gk1 = np.zeros((FSIZE, FSIZE)); gk1[FSIZE//2, FSIZE//2] = 1
    gk2 = (scipy.ndimage.gaussian_filter(gk1,sigma=.5) - scipy.ndimage.gaussian_filter(gk1,sigma=1.0))
    dog = torch.Tensor(gk2[np.newaxis,np.newaxis,:,:]).to(device) #  Adding two singleton dimensions for input and output channels (1 each)



    print("Initialization time:", time.time()-tic, "Device:", device)
    testtargets = traintargets = []; testouts = trainouts = []
    tic = time.time()
    firstpass=True
    nbbatches = 0


    # Tensors to store statistics for each epoch
   
    pre_w_grad_mean= []
    pre_w_grad_std= []
    post_w_grad_mean= []
    post_w_grad_std=[]

    coeffs_mean = []
    coeffs_std = []

    total_activations = []

    # TRAINING ///////////////////////////////////

    total_function_time = 0.0

    print_grad= False
    start_time_loop = time.time()
    times_computed =0

    for epoch in range(NBLEARNEPOCHS+2):

        current = time.time()
        elapsed = current - start_time_loop
        print(f"Epoch {epoch}, time elapsed: {elapsed}")

        myloader = testloader if epoch == NBLEARNEPOCHS +1 else trainloader
        for numbatch, (x, targets) in enumerate(myloader):
            
            current = time.time()
            elapsed = current - start_time_loop
            print(f"Epoch: {epoch}, Batch: {numbatch}, Time elapsed: {elapsed}")

            nbbatches += 1

            # Prepare the input images and decorrelate them with the difference-of-gaussian filters
            with torch.no_grad():         
                x = x.to(device); targets = targets.to(device)
                x = x - torch.mean(x, dim=(1, 2,3),keepdim=True)
                x = x / (1e-10 + torch.std(x, dim=(1, 2,3), keepdim=True)) 
                x = F.conv2d(x, dog, groups=1, padding=FSIZE//2)    # DoG filtering 


            # Now run the network's layers in succession
            
            for numl in range(NL):

                optimizers[numl].zero_grad()  # We use one optimizer per layer for added clarity, but this is not strictly necessary (with judicious use of .detach() to stop gradient flow)

                # We normalize the input
                with torch.no_grad():
                    x = x - torch.mean(x, dim=(1, 2,3),keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(1, 2,3), keepdim=True)) 

                # We apply the weight convolutions, giving us the feedforward input into each cell (and building the first part of the computational graph)
                
                prelimy = F.conv2d(x, w[numl], stride=STRIDES[numl])  
                
                # Then we compute the "real" output (y) of each cell, with winner-take-all competition
                with torch.no_grad():                
                    realy = (prelimy - thres[numl])
                    tk = torch.topk(realy.data, K[numl], dim=1, largest=True)[0]
                    realy.data[realy.data < tk.data[:,-1,:,:][:, None, :, :]] = 0       
                    realy.data = (realy.data > 0).float()

                # Then we compute the surrogate output yforgrad, whose gradient computations produce the desired Hebbian output
                # Note: We must not include thresholds here, as this would not produce the expected gradient expressions. The actual values will come from realy, which does include thresholding.

                yforgrad = prelimy - 1/2 * torch.sum(w[numl] * w[numl], dim=(1,2,3))[None,:, None, None]                # Instar rule, dw ~= y(x-w)  
                # yforgrad = prelimy - 1/2 * torch.sum(w[numl] * w[numl], dim=(1,2,3))[None,:, None, None] * realy.data # Oja's rule, dw ~= y(x-yw)
                # yforgrad = prelimy                                                                                      # Plain Hebb, dw ~= xy

                yforgrad.data = realy.data # We force the value of yforgrad to be the "correct" y

                # We perform the backward pass and the learning step, which applies the desired Hebbian updates
                loss = torch.sum( -1/2 * yforgrad * yforgrad) 
                loss.backward()

                if(firstpass):
                    # Just to check shapes are correct
                    with torch.no_grad():
                        #print(f"----- Pre optimizer step w grad. {w[numl].grad.shape}")
                        onefilter= torch.ones_like(w[numl])
                        #print(f"--- shape w: {w[numl].shape} shape x {x.shape} shape onefiter {onefilter.shape}")

                        manual=  F.conv2d(x,w[numl],stride= 1) * F.conv2d(x,onefilter,stride=1)
                        print(f"--- Manual shape: {manual.shape} vs computed {w[numl].grad.shape}")
                        if(print_grad):
                            print(f"--- Manual content: {manual.shape} vs computed {w[numl].grad.shape}")
                            print_grad= False

                if nbbatches > 100 and epoch < NBLEARNEPOCHS:  # No weight modifications before batch 100 (burn-in) or after learning epochs (during data accumulation for training / testing)
                    # Modify gradients using the coefficient vector only for the first layer
                    #if(numl==0):
                    start_time = time.time()
                    with torch.no_grad():  # Ensure no tracking history 
                        # trying parallelization for all channels:

                        t1 = time.time()
                        numf, fsize =  w[numl].shape[0], w[numl].shape[2]
                        convsize = prelimy.shape[2]
                        ch = x.shape[1]

                        coeffsMatrix= torch.zeros_like(w[numl], device= x.device)

                        for i in range(numf*fsize*fsize):

                            # --------------------------------------------------------------------
                            # Computing the current weight's alpha, beta, gamma, delta
                            index = i%(fsize*fsize)
                            # filter number
                            alpha = int((i - (index)) /(fsize*fsize) )
                            beta = int(ch)
                            #filter weight position:
                            delta = int(index% fsize)
                            gamma = int((index - delta) /fsize)
                            # --------------------------------------------------------------------
                            
                            p1 = x[:, :, gamma: gamma+convsize, delta: delta+convsize]  # x (100, 3, 33, 33) ; p1 shape (100, 3, selection,selection)
                            p2 = p1.reshape(p1.shape[1],p1.shape[0], -1)  # canales 3 columnas 100 canales de 841 valores shape (3, 100, 841)

                            q = prelimy[:, alpha, gamma, delta] #100 valores shape(3, 100)
                         
                            # --------------------------------------------------------------------

                            # Mean across the batch dimension
                            mean_x = p2.mean(dim=1, keepdim=True) # ([3, 1, 841])
                            mean_y = q.mean(dim=0, keepdim=True) # ([1])

                            # Deviations from mean
                            xm = p2 - mean_x # ([3, 100, 841]) #diff per channels
                            ym = q - mean_y # ([100])

                            # Covariance and variance 
                            cov_xy = (xm * ym.view(1,-1, 1).expand(xm.shape[0],-1,1)).mean(dim=1) #([3,100, 841]).mean = shape ([3, 841]) # FIXED
                            var_x = (xm ** 2).mean(dim=1) #shape ([3, 841])
                            var_y = (ym ** 2).mean(dim=0) #shape ([1])

                            # Pearson correlation
                            std_x = torch.sqrt(var_x)
                            std_y = torch.sqrt(var_y)

                            correlation = cov_xy / (std_x * std_y)
                    
                            # Handle NaN and infinities possibly caused by division by zero
                            correlation = torch.nan_to_num(correlation, nan=0.0)

                            coeffsMatrix[alpha,:beta,gamma,delta]= correlation.mean(dim=1)[:beta]
                            t2 = time.time()

                        #print(f"time taken: {t2-t1}")
                        #print(f"shapes of w and coeffs: {w[numl].grad.shape},{coeffsMatrix.shape}")
                                                
                        pre_w_grad_mean.append(w[numl].reshape(*w[numl].shape[:2], -1).mean(dim=2))
                        pre_w_grad_std.append( w[numl].reshape(*w[numl].shape[:2], -1).std(dim=2))

                        mask = (torch.abs(coeffsMatrix)> float(correlation_threshold))
                        sign_tensor = torch.sign(coeffsMatrix)

                        w[numl].grad[mask] *= (coeffsMatrix[mask] + sign_tensor[mask])   #Apply coefficients +/-1 to get the desired range
                            #print(f"W grad param update shape : {w[numl].grad.shape}")

                        #TODO: STORAGE of mean of coeffs per filter and channel
                        print(coeffsMatrix.shape)

                        coeffs_mean.append(torch.abs(coeffsMatrix.reshape(*coeffsMatrix.shape[:2], -1)).mean(dim=2))  # dim is (100, 1, 25x25) 
                        coeffs_std.append(torch.abs(coeffsMatrix.reshape(*coeffsMatrix.shape[:2], -1)).std(dim=2))

                        #post_w_grad_mean.append(w[numl].reshape(*w[numl].shape[:2], -1).mean(dim=2))
                        #post_w_grad_std.append(w[numl].reshape(*w[numl].shape[:2], -1).std(dim=2))

                        activations = torch.sum(mask)
                        total_activations.append(activations/mask.numel()) # percentage of synapses activated

                            
                    end_time = time.time()
                        
                    duration = end_time - start_time
                    total_function_time += duration

                optimizers[numl].step()              
                w[numl].data =  w[numl].data / (1e-10 + torch.sqrt(torch.sum(w[numl].data ** 2, dim=[1,2,3], keepdim=True)))  # Weight normalization

                # We show the sizes of the layers. Watch especially the size of the last layer - if it's just 1x1, there won't be a lot of information there.
                #if firstpass:
                    #print("Layer", numl, ": x.shape:", x.shape, "y.shape (before MaxP):", realy.shape, end=" ")

                # Apply pooling to produce the input of the next layer (or the final network output)
                with torch.no_grad():
                    x = F.avg_pool2d(realy, POOLDIAMS[numl], stride=POOLSTRIDES[numl]).detach()       

                    #if firstpass:
                        #print("y.shape (after MaxP):", x.shape)  # The layer's final output ("y") is now x for the next step

                    # Threshold adaptation is based on realy, i.e. the one used for plasticity. Always binarized (firing vs. not firing).
                    thres[numl] +=  MUTHRES *   (torch.mean((realy.data > 0).float(), dim=(0,2,3))[None, :, None, None] -  TARGETRATE[numl])


            # After all layers are done

            if epoch >= NBLEARNEPOCHS:  # Collecting data to train and test a linear classifier, based on (frozen) network response to the training and testing datasets, respectively.
                # We simply collect the outputs of the network, as well as the labels. The actual training/testing will occur below, with a linear classifier.
                if epoch == NBLEARNEPOCHS:
                    testtargets.append(targets.data.cpu().numpy())
                    testouts.append(x.data.cpu().numpy())
                elif epoch ==NBLEARNEPOCHS + 1:
                    traintargets.append(targets.data.cpu().numpy())
                    trainouts.append(x.data.cpu().numpy())
                else:
                    raise ValueError("Too many epochs!")

            firstpass = False 


        # After all batches for this epoch are done

        #if nbbatches % 1000 == 0: 
        print("Number of batches after epoch", epoch,  ":", nbbatches, "- time :", (time.time()-tic), "s")



    print("Training done..")


    print(f'Total time spent in Pearson Correlation: {total_function_time:.2f} seconds')

    
    # END OF TRAINING: save tensors for testing ///////////////////////////////////
    results = {
        'testtargets': testtargets,
        'testouts': testouts,
        'traintargets': traintargets,
        'trainouts': trainouts,

        'post_w_grad_mean': post_w_grad_mean,
        'post_w_grad_std':post_w_grad_std, 
        'pre_w_grad_mean':pre_w_grad_mean,
        'pre_w_grad_std':pre_w_grad_std,

        "coeffs_mean": coeffs_mean,
        "coeffs_std": coeffs_std,

        "total_activations": total_activations 
    }

    # Get the current date and time

    # Add 8 hours to the current time to get SG time
    current_time = datetime.now() + timedelta(hours=8)

    current_time = current_time.strftime("%Y%m%d_%H%M%S")
    output_dir = './results/PearsonBatch04_05/fixedCoeff/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the filename by appending the current date and time
    file_name = f'allBatches{layers}LayersCoeffs{correlation_threshold}_{current_time}.pth'
    file_path = os.path.join(output_dir, file_name)

     #Save the tensor to the file
    torch.save(results, file_path)


if __name__ == '__main__':
    main()


