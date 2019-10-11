
import fire

def mura_experiment(seed, body_part, n_runs, gpu_id, n_shot=None, train_cutout=False, train_apga=False, train_gradcam=False, train_end2end=False):

    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)





    import torch
    from torch import nn
    import numpy as np
    import torchvision
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import ToTensor, Normalize, Compose
    from torch.nn import functional as F
    import torch.optim as optim
    from torch.autograd import Variable


    device = torch.device('cuda')


    # Imports


    from matplotlib import pyplot as plt
    from pylab import imshow
    from tqdm import tqdm
    import time

    import pandas as pd
    import numpy as np
    from skimage.transform import resize, rescale
    from skimage import exposure

    from model_utils import random_eraser
    from grad_cam import GradCAM
    from scipy import ndimage

    from TernausNet.unet_models import unet11
    



    def cutout_img(img, p=0.5):
        if np.random.random() < p:
            img = img.cpu().numpy()
            img_one_channel = img[0]
            mask = np.where(np.random.normal(size=(224, 224))<0.1, 0, img_one_channel)
            img = np.stack((mask,)*n_channels, -1)
            img = ToTensor()(img).to(dtype=torch.float)
        return img



    cutout = random_eraser.get_random_eraser(p=1, s_l=0, s_h=224, 
                                    r_1=0.3, r_2=1/0.3,
                                    v_l=0, v_h=0, pixel_level=True)



    def cutout_img(img, p=0.5):
        if np.random.random() < p:
            
            img = cutout(np.moveaxis(img.cpu().numpy(), 0, -1))
            #img = np.stack((img,)*n_channels, -1).astype(np.float32)
            img = ToTensor()(img).to(dtype=torch.float)
        return img



    def show_cutout(idx):
        debug_img, _ = accession_train_dataset.__getitem__(idx)
        debug_img = torch.tensor(debug_img)
        plt.figure()
        print(cutout_img(debug_img, p=1).numpy().shape)
        plt.imshow(np.moveaxis(cutout_img(debug_img, p=1).numpy(), 0, -1))
        plt.show()


    # ## Training methods for baseline

    device = torch.device('cuda')
    
    def get_acc(model, test_generator):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_generator)):
                inputs, labels = data
                #inputs = np.moveaxis(inputs, -1, 1)
                #inputs = torch.autograd.Variable(torch.from_numpy(inputs).float()).cuda()
                #l = torch.from_numpy(labels).long().cuda()
                #labels = torch.autograd.Variable(torch.from_numpy(labels).long()).cuda()
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = model(inputs)
                #print(outputs.data)
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                total += labels.size(0)
                #print(total)
                correct += (predicted == labels).sum().item()
                #print(predicted)

        print('Accuracy : ',(100 * correct / total), '%')
        return 100 * correct / total


    def train(model, n_epoch, optimizer, loss_f, train_generator, val_generator, save_path, aug=False, patience=10):
        best_val_acc = 0
        patience = patience
        if patience:
            running_patience = patience
            prev_flag = False
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            
            if patience:
                if running_patience <= 0:
                    break
                    
            for i, data in tqdm(enumerate(train_generator)):
                
                
                # get the inputs
                inputs, labels = data
                
                
                
                inputs = inputs.to(device=device, dtype=torch.float)
                
                labels = labels.to(device=device, dtype=torch.long)
                
                aug = aug
                if aug:
                    
                    aug_imgs = Variable(torch.Tensor())
                    #aug_labels = Variable(torch.Tensor()).to(device=device, dtype=torch.long)
                    for j, img in enumerate(inputs):
                        #inputs[i] = mask_aug(policy, img, p=0.5)
                        #if labels[i] > 0:
                        #aug_imgs = torch.cat([aug_imgs, mask_aug(policy, img, p=1).unsqueeze(0)], dim=0)
                        #aug_labels = torch.cat([aug_labels, labels[j].unsqueeze(0)], dim=0)
                        #inputs[i] = gm_masked_img(img, p=1)
                        #aug_imgs = torch.cat([aug_imgs, gm_masked_img(img, p=1).unsqueeze(0)], dim=0)
                        aug_imgs = torch.cat([aug_imgs, cutout_img(img, p=1).unsqueeze(0)], dim=0)

                    #plt.imshow(np.moveaxis(inputs[1].cpu().numpy(), 0, -1))
                    #plt.show()
                    #return

                    aug_imgs = aug_imgs.to(device=device, dtype=torch.float)
                    #aug_labels = aug_labels.to(device=device, dtype=torch.long)
                

                #'''
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                
                loss.backward()
                optimizer.step()
                #'''
                
                if aug:
                    #batch_acc = count_acc(outputs, labels)
                    if aug_imgs.dim() > 1:
                        # for augs
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = model(aug_imgs)
                        aug_loss = loss_f(outputs, labels)
                        aug_loss.backward()
                        optimizer.step()

                # print statistics
                
                if aug:
                    running_loss += (loss.item() + aug_loss.item()) / 2
                else:
                    running_loss += loss.item()
                #print(i, end=" ")

            print('training loss: ', (epoch + 1, running_loss / len(train_generator)))
            print('Validation:')
            val_acc = get_acc(model, val_generator)
            print('prev_best_val:', best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print('saved!')
                if patience:
                    prev_flag = False
                    running_patience = patience
            else:
                if patience:
                    print('Not better val')
                    if prev_flag:
                        running_patience -= 1
                        print('Patience left: ', running_patience)
                    prev_flag = True
                    
            running_loss = 0.0
            
        return best_val_acc



    # ## Policy Gradient


    def predict_masks(unet, dataloader):
        #unet.eval()
        masks = []
        log_masks = Variable(torch.Tensor())
        for i, data in enumerate(dataloader):
            img, label = data
            #with torch.no_grad():
                #input_img = torch.unsqueeze(ToTensor()(img).to(device, dtype=torch.float), dim=0)
                #input_img = (ToTensor()(img).to(device, dtype=torch.float))
            img = (img).to(device, dtype=torch.float)
            #with torch.no_grad():
                
            mask = (unet(img))
            mask_array = (mask.data)[0].cpu().numpy()[0]
            img = (img.data)[0].cpu().numpy()[0]
            #img = np.moveaxis(img, 0, -1)
            masked_img = masking_img(img, mask_array)
            masks.append(masked_img)
            if log_masks.dim() != 0:
                log_masks = torch.cat([log_masks, torch.log((mask))])
            else:
                log_masks = torch.log((mask))
            
        return masks, log_masks

    def predict_masks_from_imgs(policy, img,
                               morphology_cleaning=False, largest_connected=False, fill_enclosed=False, 
                                mask_action=False):
        with torch.no_grad():
            #input_img = torch.unsqueeze(ToTensor()(img).to(device, dtype=torch.float), dim=0)
            #input_img = (ToTensor()(img).to(device, dtype=torch.float))
            img = torch.unsqueeze((img).to(device, dtype=torch.float), dim=0)
        with torch.no_grad():
            mask = (policy(img))
            #print(mask)
            #print(mask*(mask>0.5).to(dtype=torch.float))
        mask_array = (mask.data)[0].cpu().numpy()[0]
        img = (img.data)[0].cpu().numpy()[0]
        #img = np.moveaxis(img, 0, -1)
        masked_img = masking_img(img, mask_array, 
                                 morphology_cleaning=morphology_cleaning, 
                                 largest_connected=largest_connected, 
                                 fill_enclosed=fill_enclosed, 
                                 mask_action=mask_action)
        
        log_mask = torch.log(mask)
        #print((log_mask<0 )
        
        return ToTensor()(masked_img), mask_array
            
    def masking_img(img, mask, 
                    morphology_cleaning=False, largest_connected=False, fill_enclosed=False, 
                    mask_action=False):
        #mask_array = np.stack((mask_array,)*n_channels, -1)
        
        if mask_action:
            binary_mask = mask>0.5
        else:
            binary_mask = mask<0.5
        
        binary_mask = post_processing(binary_mask, morphology_cleaning, largest_connected, fill_enclosed) 
        
        output = np.where(binary_mask, 0, img)
        
        output = np.stack((output,)*n_channels, -1)
        return output



    def post_processing(binary_img, morphology_cleaning, largest_connected, fill_enclosed):
        if morphology_cleaning:
            # Remove small white regions
            binary_img = ndimage.binary_opening(binary_img)
            # Remove small black hole
            binary_img = ndimage.binary_closing(binary_img)
        if largest_connected:
            label_im, nb_labels = ndimage.label(binary_img)
            sizes = ndimage.sum(binary_img, label_im, range(nb_labels + 1))
            max_size = np.max(sizes)
            mask_size = sizes < max_size
            remove_pixel = mask_size[label_im]
            binary_img[remove_pixel] = False
        if fill_enclosed:
            binary_img = ndimage.morphology.binary_fill_holes(binary_img)
        return binary_img.astype('float64')


    def mask_aug(policy, img, p=0.5, mask_action=False):
        if np.random.random() < p:
            img, log_mask = predict_masks_from_imgs(policy, img, mask_action=mask_action)
            #policy.policy_history = torch.cat([policy.policy_history, log_mask])
        return img


    def gm_masked_img(img, p=0.5):
        if np.random.random() < p:
            img = img.cpu().numpy()
            img_one_channel = img[0]
            mask = np.where(gaussian_mixture(img_one_channel)==False, 0, img_one_channel)
            img = np.stack((mask,)*n_channels, -1)
            img = ToTensor()(img).to(dtype=torch.float)
        return img



    def show_gm_mask(idx):
        debug_img, label = accession_train_dataset.__getitem__(idx)
        debug_img = torch.tensor(debug_img)
        plt.figure()
        plt.imshow(np.moveaxis(gm_masked_img(debug_img, p=1).numpy(), 0, -1))
        plt.show()
        return label


    def show_debug_mask(idx, save=True):
        debug_img, label = accession_train_dataset.__getitem__(idx)
        debug_mask, mask = predict_masks_from_imgs((policy), ToTensor()(np.moveaxis(debug_img, 0, -1)))
        if save:
            mask = np.moveaxis(mask, -1, 0)
            show_and_save(debug_mask.numpy(), 
                      'policy_gradient_pngs/'+'masked_hip_'+str(i)+'_class_'+str(label)+'.png')
            show_and_save((mask>0.5), 
                      'policy_gradient_pngs/'+'hip_mask_'+str(i)+'_class_'+str(label)+'.png', 
                          cmap=plt.get_cmap('gray'))
        else:
            plt.figure()
            plt.imshow(np.moveaxis(debug_mask.numpy(), 0, -1))
            plt.show()
        return label



    def show_debug_mask_prob(idx):
        debug_img, label = accession_train_dataset.__getitem__(idx)
        plt.figure()
        with torch.no_grad():
            #input_img = torch.unsqueeze(ToTensor()(img).to(device, dtype=torch.float), dim=0)
            #input_img = (ToTensor()(img).to(device, dtype=torch.float))
            img = torch.unsqueeze(torch.tensor(debug_img).to(device, dtype=torch.float), dim=0)
        with torch.no_grad():
            mask = (policy(img))
            #print(mask)
            #print(mask*(mask>0.5).to(dtype=torch.float))
        mask_array = (mask.data)[0].cpu().numpy()
        #debug_mask, _ = predict_masks_from_imgs((policy), ToTensor()(np.moveaxis(debug_img, 0, -1)))
        print(mask_array.shape)
        plt.imshow(np.moveaxis(mask_array[0], 0, -1), cmap='jet')
        plt.show()
        return label



    def show_val_debug_mask(idx):
        debug_img, label = accession_val_dataset.__getitem__(idx)
        debug_mask, _ = predict_masks_from_imgs((policy), ToTensor()(np.moveaxis(debug_img, 0, -1)))
        show_and_save(debug_mask.numpy(), 
                      'policy_gradient_pngs/'+'masked_hip_val_'+str(i)+'_class_'+str(label)+'.png')
        
        return label


    def show_debug_img(idx):
        debug_img, label = accession_train_dataset.__getitem__(idx)
        show_and_save(debug_img, 
                      'policy_gradient_pngs/'+'hip_'+str(i)+'_class_'+str(label)+'.png')
        return label



    def show_and_save(img, save_path, cmap=None):
        fig = plt.figure()
        fig.set_size_inches(1, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        if cmap:
            plt.imshow(np.moveaxis(img, 0, -1), cmap=cmap)
        else:
            plt.imshow(np.moveaxis(img, 0, -1))
        plt.savefig(save_path, dpi=224)
        plt.show()
        plt.close()


    def count_acc(outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        #print(pred/icted)
        total = labels.size(0)
        #print(total)
        correct = (predicted == labels).sum().item()
        acc = correct / total * 100
        return acc


    def get_acc_with_PG(model, test_generator):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_generator)):
                inputs, labels = data
                
                for i, img in enumerate(inputs):
                    inputs[i] = mask_aug(policy, img, p=1)
                    
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()


        print('Accuracy : ',(100 * correct / total), '%')
        return 100 * correct / total




    def get_acc_with_PG_tta(model, test_generator):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_generator)):
                inputs, labels = data

                og_inputs = inputs.to(device=device, dtype=torch.float)
                outputs = model(og_inputs)
                for i, img in enumerate(inputs):
                    inputs[i] = mask_aug(policy, img, p=1)
                    
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                aug_outputs = model(inputs)
                
                outputs = F.softmax(outputs)
                aug_outputs = F.softmax(aug_outputs)

                outputs = (outputs + aug_outputs) / 2
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()


        print('TTA Accuracy : ',(100 * correct / total), '%')
        return 100 * correct / total



    def train_aug_step(model, optimizer, loss_f, inputs, labels):
        aug_imgs = Variable(torch.Tensor())
        aug_labels = Variable(torch.Tensor()).to(device=device, dtype=torch.long)

        for i, img in enumerate(inputs):
            #inputs[i] = mask_aug(policy, img, p=0.5)
            #if labels[i] > 0:
            #aug_imgs = torch.cat([aug_imgs, mask_aug(policy, img, p=1, mask_action=False).unsqueeze(0)], dim=0)
            aug_labels = torch.cat([aug_labels, labels[i].unsqueeze(0)], dim=0)

            aug_imgs = torch.cat([aug_imgs, cutout_img(img, p=1).unsqueeze(0)], dim=0)

        aug_imgs = aug_imgs.to(device=device, dtype=torch.float)
        aug_labels = aug_labels.to(device=device, dtype=torch.long)

        #'''
        # for augs
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(aug_imgs)
        aug_loss = loss_f(outputs, aug_labels)
        aug_loss.backward()
        optimizer.step()
        
        return aug_loss


    def train_with_PG(policy, model, n_epoch, optimizer, pg_optimizer, loss_f, 
                      train_generator, val_generator, 
                      save_path, policy_save_path, best_val_acc=0, aug=True, cutout=False, patience=10):
        best_val_acc = best_val_acc
        if patience:
            running_patience = patience
            prev_flag = False
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            policy.train()
            
            if patience:
                if running_patience <= 0:
                    break
            
            
            #show_debug_mask(1)
            
            policy.loss_history = []
            for i, data in tqdm(enumerate(train_generator)):
                model.train()
                
                
                # get the inputs
                inputs, labels = data
                
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                
                #'''
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_f(outputs, labels)

                loss.backward()
                optimizer.step()
                #'''
                    
                #policy.eval()
                aug = aug
                if aug:
                    aug_imgs = Variable(torch.Tensor())
                    aug_labels = Variable(torch.Tensor()).to(device=device, dtype=torch.long)
                    for j, img in enumerate(inputs):

                        aug_imgs = torch.cat([aug_imgs, mask_aug(policy, img, p=1, mask_action=True).unsqueeze(0)], dim=0)
                        aug_labels = torch.cat([aug_labels, labels[j].unsqueeze(0)], dim=0)

     
                    aug_imgs = aug_imgs.to(device=device, dtype=torch.float)
                    aug_labels = aug_labels.to(device=device, dtype=torch.long)

                    if aug_imgs.dim() > 1:

                        model.eval()
                        aug_outputs = model(aug_imgs)
                        aug_loss = loss_f(aug_outputs, aug_labels)

                        loss_reward = (aug_loss-loss) # we want a higher aug loss for adversarial

                        update_policy(policy, pg_optimizer, loss_reward, val_acc=None, images = inputs)

                        model.train()
                        
                        aug_imgs = Variable(torch.Tensor())
                        aug_labels = Variable(torch.Tensor()).to(device=device, dtype=torch.long)
                        
                        for k, img in enumerate(inputs):
                            #inputs[i] = mask_aug(policy, img, p=0.5)
                            #if labels[i] > 0:
                            aug_imgs = torch.cat([aug_imgs, mask_aug(policy, img, p=1, mask_action=False).unsqueeze(0)], dim=0)
                            aug_labels = torch.cat([aug_labels, labels[k].unsqueeze(0)], dim=0)
                            

                        aug_imgs = aug_imgs.to(device=device, dtype=torch.float)
                        aug_labels = aug_labels.to(device=device, dtype=torch.long)
                        
                        #'''
                        # for augs
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = model(aug_imgs)
                        aug_loss = loss_f(outputs, aug_labels)
                        aug_loss.backward()
                        optimizer.step()
                        #'''
                        
                        if cutout:
                            cutout_loss = train_aug_step(model, optimizer, loss_f, inputs, labels)
                
                #print(i)
                if i > 0 and i % 100 == 0:
                    print('Validation:')
                    val_acc = get_acc(model, val_generator)
                    print('prev_best_val:', best_val_acc)
                    print('Policy Loss:', np.mean(policy.loss_history))

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), save_path)

                        torch.save(policy.state_dict(), policy_save_path)

                        print('saved!')
                        if patience:
                            prev_flag = False
                            running_patience = patience
                    else:
                        if patience:
                            print('Not better val')
                            if prev_flag:
                                running_patience -= 1
                                print('Patience left: ', running_patience)
                            prev_flag = True
                

                
                #loss = (aug_loss+loss) / 2
                #loss = aug_loss

                # print statistics
                running_loss += loss.item()
                #print(i, end=" ")

            print('training loss: ', (epoch + 1, running_loss / len(train_generator)))
            print('Validation:')
            val_acc = get_acc(model, val_generator)
            print('prev_best_val:', best_val_acc)
            print('Policy Loss:', np.mean(policy.loss_history))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                
                torch.save(policy.state_dict(), policy_save_path)
                
                print('saved!')
                if patience:
                    prev_flag = False
                    running_patience = patience
            else:
                if patience:
                    print('Not better val')
                    if prev_flag:
                        running_patience -= 1
                        print('Patience left: ', running_patience)
                    prev_flag = True
                    
                    
                    
                    
            tta=True    
            if tta:
                val_acc = get_acc_with_PG_tta(model, val_generator)
                #print('prev_best_val:', best_val_acc)
                #print('Policy Loss:', np.mean(policy.loss_history))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), save_path)

                    torch.save(policy.state_dict(), policy_save_path)

                    print('saved!')
                else:
                    print('Not better val')
                    
                    
                    
                    
                    
                    
            running_loss = 0.0
            
            #for i in range(10, 20):
                #show_debug_mask(i)
                
        return round(best_val_acc, 4)


    def train_end2end(policy, model, n_epoch, optimizer, pg_optimizer, loss_f, 
                      train_generator, val_generator, 
                      save_path, policy_save_path, best_val_acc=0, 
                      aug=True, cutout=False, patience=10, aug_reg=False):
        sim_loss = nn.BCELoss()
        if patience:
            running_patience = patience
            prev_flag = False
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            policy.train()

            if patience:
                if running_patience <= 0:
                    break


            #show_debug_mask(1)

            policy.loss_history = []
            for i, data in tqdm(enumerate(train_generator)):
                model.train()


                # get the inputs
                inputs, labels = data

                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)

                #'''
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_f(outputs, labels)

                loss.backward()
                optimizer.step()
                #'''

                #policy.eval()

                if aug:
                    masks = policy(inputs)

                    #probs = torch.add(masks,other=torch.tensor(np.random.normal(0, 0.1, size=masks.shape)).to(device=device, dtype=torch.float))
                    probs = masks
                    #binary_masks = (probs>0.5).to(dtype=torch.float)
                    aug_imgs = masks * inputs

                    aug_outputs = model(aug_imgs)
                    aug_loss = loss_f(aug_outputs, labels)
                    
                    if aug_reg:
                            zeros = torch.zeros_like(probs)
                            zero_loss = sim_loss(probs, zeros)
                            
                            ones = torch.ones_like(probs)
                            one_loss = sim_loss(probs, ones)
                            
                            aug_loss += zero_loss * 0.5 + one_loss*0.6
                            
                    aug_loss.backward()
                    optimizer.step()

                    pg_optimizer.step()
                    
                    #train with aug images
                    aug_imgs = Variable(torch.Tensor())
                    aug_labels = Variable(torch.Tensor()).to(device=device, dtype=torch.long)

                    for k, img in enumerate(inputs):
                        #inputs[i] = mask_aug(policy, img, p=0.5)
                        #if labels[i] > 0:
                        aug_imgs = torch.cat([aug_imgs, mask_aug(policy, img, p=1, mask_action=False).unsqueeze(0)], dim=0)
                        aug_labels = torch.cat([aug_labels, labels[k].unsqueeze(0)], dim=0)


                    aug_imgs = aug_imgs.to(device=device, dtype=torch.float)
                    aug_labels = aug_labels.to(device=device, dtype=torch.long)

                    #'''
                    # for augs
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = model(aug_imgs)
                    aug_loss = loss_f(outputs, aug_labels)
                    aug_loss.backward()
                    optimizer.step()


                    if cutout:
                        cutout_loss = train_aug_step(model, optimizer, loss_f, inputs, labels)

                #print(i)
                if i > 0 and i % 100 == 0:
                    print('Validation:')
                    val_acc = get_acc(model, val_generator)
                    print('prev_best_val:', best_val_acc)
                    #print('Policy Loss:', np.mean(policy.loss_history))

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), save_path)

                        torch.save(policy.state_dict(), policy_save_path)

                        print('saved!')
                        if patience:
                            prev_flag = False
                            running_patience = patience
                    else:
                        if patience:
                            print('Not better val')
                            if prev_flag:
                                running_patience -= 1
                                print('Patience left: ', running_patience)
                            prev_flag = True

                show_debug_mask_mura(1, elbow_val_dataset, save=False)

                #loss = (aug_loss+loss) / 2
                #loss = aug_loss

                # print statistics
                running_loss += loss.item()
                #print(i, end=" ")

            print('training loss: ', (epoch + 1, running_loss / len(train_generator)))
            print('Validation:')
            val_acc = get_acc(model, val_generator)
            print('prev_best_val:', best_val_acc)
            print('Policy Loss:', np.mean(policy.loss_history))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)

                torch.save(policy.state_dict(), policy_save_path)

                print('saved!')
                if patience:
                    prev_flag = False
                    running_patience = patience
            else:
                if patience:
                    print('Not better val')
                    if prev_flag:
                        running_patience -= 1
                        print('Patience left: ', running_patience)
                    prev_flag = True




            tta=True    
            if tta:
                val_acc = get_acc_with_PG_tta(model, val_generator)
                #print('prev_best_val:', best_val_acc)
                #print('Policy Loss:', np.mean(policy.loss_history))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), save_path)

                    torch.save(policy.state_dict(), policy_save_path)

                    print('saved!')
                else:
                    print('Not better val')


            running_loss = 0.0

            #for i in range(10, 20):
                #show_debug_mask(i)

        return round(best_val_acc, 4)


    # In[53]:

    def train_gradcam(model, gradcam_model, n_epoch, optimizer, loss_f, train_generator, val_generator, save_path, gradcam=False, aug=False, patience=10):
        best_val_acc = 0

        if gradcam:
            gcam = GradCAM(model=gradcam_model)

        if patience:
            running_patience = patience
            prev_flag = False
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            
            if patience:
                if running_patience <= 0:
                    break
                    
            for i, data in tqdm(enumerate(train_generator)):
                
                
                # get the inputs
                o_inputs, o_labels = data
                
                
                
                inputs = o_inputs.to(device=device, dtype=torch.float)
                
                labels = o_labels.to(device=device, dtype=torch.long)
                
                if gradcam:

                    probs, ids = gcam.forward(o_inputs.to(device=device, dtype=torch.float))
                    gcam.backward(ids=ids[:, [0]])
                    gc_image = gcam.generate(target_layer='features.norm5')

                    aug_imgs = o_inputs.to(device=device, dtype=torch.float) * (gc_image > 0.5).to(dtype=torch.float)

                    if aug_imgs.dim() > 1:
                        # for augs
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = model(aug_imgs)
                        aug_loss = loss_f(outputs, labels)
                        aug_loss.backward()
                        optimizer.step()

                if aug:
                    
                    aug_imgs = Variable(torch.Tensor())
                    for j, img in enumerate(inputs):
                        aug_imgs = torch.cat([aug_imgs, cutout_img(img, p=1).unsqueeze(0)], dim=0)


                    aug_imgs = aug_imgs.to(device=device, dtype=torch.float)
                

                #'''
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                
                loss.backward()
                optimizer.step()
                #'''
                
                if aug:
                    #batch_acc = count_acc(outputs, labels)
                    if aug_imgs.dim() > 1:
                        # for augs
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = model(aug_imgs)
                        aug_loss = loss_f(outputs, labels)
                        aug_loss.backward()
                        optimizer.step()

                # print statistics
                
                if aug:
                    running_loss += (loss.item() + aug_loss.item()) / 2
                else:
                    running_loss += loss.item()
                #print(i, end=" ")

            print('training loss: ', (epoch + 1, running_loss / len(train_generator)))
            print('Validation:')
            val_acc = get_acc(model, val_generator)
            print('prev_best_val:', best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print('saved!')
                if patience:
                    prev_flag = False
                    running_patience = patience
            else:
                if patience:
                    print('Not better val')
                    if prev_flag:
                        running_patience -= 1
                        print('Patience left: ', running_patience)
                    prev_flag = True
                    
            running_loss = 0.0
            
        return best_val_acc

    class Policy(nn.Module):
        def __init__(self):
            super(Policy, self).__init__()
            
            self.gamma = gamma
            self.baseline = None
            
            # Episode policy and reward history 
            self.policy_history = Variable(torch.Tensor()).to(device=device)
            self.reward_episode = []
            # Overall reward and loss history
            self.reward_history = []
            self.loss_history = []
            
            self.unet = unet11(pretrained='carvana')
            self.prediction = nn.Sigmoid()

            
            
        def forward(self, x):
            x = self.unet(x)
            x = self.prediction(x)
            return x

    def update_policy(policy, pg_optimizer, loss_reward, val_acc=None, images=None, sim_loss_fn=nn.BCELoss()):
        loss_reward = loss_reward.item()
        #loss_reward = val_acc / 100
        
        if policy.baseline is None:
            policy.baseline = loss_reward
        else:
            decay = 0.50
            policy.baseline = decay * policy.baseline + (1 - decay) * loss_reward
            

        masks = policy(images)
        probs = masks
        binary_masks = (probs>0.5).to(device=device, dtype=torch.float)
        
        sim_loss = sim_loss_fn(probs, binary_masks)
        zeros = torch.zeros_like(probs)
        zero_loss = sim_loss_fn(probs, zeros)
        
        ones = torch.ones_like(probs)
        one_loss = sim_loss_fn(probs, ones)

        R = (loss_reward - policy.baseline)
        loss = ((sim_loss) * R)*100 + zero_loss*torch.abs(R)*0.1 + one_loss*torch.abs(R)*0.01# REINFORCE loss + penalty for masking too much
        
        #print('REINFORCE loss, ', loss*10)

        # Update network weights
        pg_optimizer.zero_grad()


        loss.backward(retain_graph=False)
        pg_optimizer.step()

        #Save and intialize episode history counters
        policy.loss_history.append(loss.item())
        policy.reward_history.append(loss_reward)



    # ## MURA

    # In[72]:


    from skimage import io, transform
    import PIL
    from PIL import Image


    # In[73]:

    baseline_accs = []
    cutout_accs = []
    pg_accs = []
    end2end_accs = []
    gradcam_accs = []

    model = torchvision.models.densenet169(pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(1664, 2))
    model = model.to(device=device)

    gradcam_model = torchvision.models.densenet169(pretrained=False)
    gradcam_model.classifier = nn.Sequential(nn.Linear(1664, 2))
    gradcam_model = gradcam_model.to(device=device)

    for n in range(n_runs):

        seed+=1

        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)


        class MURADataset(Dataset):
            'Generates data for Keras'
            def __init__(self, 
                         mura_frame, 
                         root_path,
                         body_part,
                         multi_body_parts=None,
                         task_size=500,
                         batch_size=32, 
                         dim=(1000, 1000), 
                         n_channels=1, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         mask=None,
                         tscl=False,
                         data_distribution=None,
                        transform=None,
                        k_shot=None):
                
                'Initialization'
                #mura_frame = mura_frame.reset_index(drop=True)
                self.multi_npy_paths = []
                self.multi_labels = []
                mura_frame = mura_frame.rename(columns = {mura_frame.columns[0]:'path'}, inplace = False)
                if not body_part:
                    self.npy_paths = mura_frame['path'].reset_index(drop=True)
                    self.labels = self.npy_paths.apply(self.path_to_label).reset_index(drop=True)
                elif not multi_body_parts:
                    self.npy_paths = mura_frame['path'][mura_frame.path.apply(lambda s: body_part in s)].reset_index(drop=True)
                    self.labels = self.npy_paths.apply(self.path_to_label).reset_index(drop=True)
                else:
                    for bodypart in multi_body_parts:
                        paths = mura_frame['path'][mura_frame.path.apply(lambda s: bodypart in s)].reset_index(drop=True)
                        labels = paths.apply(self.path_to_label).reset_index(drop=True)
                        paths, labels = np.random.choice(zip(paths, labels), task_size, replace=False)
                        self.multi_npy_paths.append(paths)
                        self.multi_labels.append(labels)
                        
                        
                self.k_shot = k_shot
                if k_shot:
                    pos = self.labels == 1
                    neg = self.labels == 0
                    self.positive_paths = self.npy_paths[pos].reset_index(drop=True)
                    self.negative_paths = self.npy_paths[neg].reset_index(drop=True)
                    self.positive_labels = self.labels[pos].reset_index(drop=True)
                    self.negative_labels = self.labels[neg].reset_index(drop=True)
                    
                    self.pos_sample_indexes = np.random.choice(np.arange(len(self.positive_paths)), size=self.k_shot, replace=False)
                    self.neg_sample_indexes = np.random.choice(np.arange(len(self.negative_paths)), size=self.k_shot, replace=False)
                    
                    pos_paths = [self.positive_paths[i] for i in self.pos_sample_indexes]
                    pos_labels = [self.positive_labels[i] for i in self.pos_sample_indexes]
                    neg_paths = [self.negative_paths[i] for i in self.neg_sample_indexes]
                    neg_labels = [self.negative_labels[i] for i in self.neg_sample_indexes]
                    self.npy_paths = (pos_paths + neg_paths)
                    self.labels = np.array(pos_labels + neg_labels)
                    #X = np.array([self.image_to_array(k) for k in X])
                        
                
                self.root_path = root_path
                self.dim = dim
                self.batch_size = batch_size
                self.n_channels = n_channels
                self.n_classes = n_classes
                self.shuffle = shuffle
                self.data_aug_funcs = data_aug_funcs
                self.normalize = normalize
                self.mask = mask
                self.data_distribution = data_distribution
                self.index_to_sample = None
                self.tscl = tscl
                #self.on_epoch_end()
                self.indexes = np.arange(len(self.labels))
                np.random.shuffle(self.indexes)
                if tscl:
                    self.tscl = TSCL(train_generator=self,        
                                            window_size=10, 
                                            thompson=True, 
                                            absolute=True)
                    
                self.transform = transform
                

            def __len__(self):
                'Denotes the number of batches per epoch'
                #return int(np.ceil(len(self.labels) / self.batch_size))
                return len(self.labels)
            
            def __getitem__(self, idx):
                image = self.index_to_image(idx)
                
                image = image.astype('float64')
                image = np.moveaxis(image, -1, 0)
                
                if self.transform:
                    image = self.transform(image)
                #image = ToTensor()(image)
                #image = image.to(device=device, dtype=torch.float)
                return image, self.index_to_label(idx)

            def __getitembatch__(self, index):
                'Generate one batch of data'
                #print(index)
                if isinstance(self.tscl, TSCL):
                    #turn fit_generator shuffle to False
                    index = self.tscl.produce_index_to_sample(index)
                    print('index to sample')
                    print(index)
                
                '''
                if self.data_distribution:
                    self.indexes = self.data_distribution #hack: generator is called before callback
                    print(self.indexes)
                '''
                
                # Generate indexes of the batch
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

                # Find list of IDs
                npy_paths_batch = [self.npy_paths[k] for k in indexes]
                labels_batch = [self.labels[k] for k in indexes]

                # Generate data
                #X, y = self.__data_generation(list_IDs_temp)
                
                X = [self.image_to_array(k) for k in npy_paths_batch]
                #print(len(X))
                y = np.array(labels_batch)
                
                
                if self.data_aug_funcs:
                    x_aug_list = []
                    for i, npy in enumerate(X):
                        x_aug_list += self.__data_augmentation(npy)
                        labels_to_add = [y[i]] * len(self.data_aug_funcs)
                        y = np.append(y, labels_to_add)
                        #print(i)
                    X += x_aug_list
                    
                X = np.stack(X)
                
                #print(X.shape)
                    
                return X, y

            def on_epoch_end(self):
                'Updates indexes after each epoch'
                #print('generator epoch end!')
                self.indexes = np.arange(len(self.labels))
                if self.shuffle == True:
                    np.random.shuffle(self.indexes)
                    
            def set_index_to_sample(self, index):
                self.index_to_sample = index
                    
            def index_to_image(self, i):
                path = self.npy_paths[i]
                return self.image_to_array(path)
            
            def index_to_label(self, i):
                return self.labels[i]
            
            def image_to_array(self, img_path):
                #try:
                #img = imread('/working/oaidl1/xray_mri/'  + img_path, flatten=False)
                #return imresize(img, (224, 224, 3))
                img = Image.open(self.root_path  + img_path) 
                img = img.resize(self.dim)
                img = np.array(img)
                #img = img[:,:,0]
                if self.mask:
                    img = self.mask(img)
                if len(img.shape) > 2:
                    img = img[:,:,0]
                img = np.stack((img,)*self.n_channels, -1)
                if self.normalize:
                    img = img / 255.
                #img = np.asarray(img)
                #if len(img.shape) == 2:
                #   img = np.stack((img,)*n_channel, -1)
                return img
                #except Exception as e:
                    #return self.image_to_array(img_path)

            def path_to_label(self, path):
                if 'positive' in path:
                    return 1
                return 0

            def set_data_distribution(p):
                self.data_distribution = p
                    
            def __path_to_npy(self, path):
                npy = np.load(self.root_path+path)
                npy = resize(npy, self.dim)
                npy = np.stack((npy,)*self.n_channels, -1)
                return npy
            
            def __data_augmentation(self, npy):
                npys = []
                for func in self.data_aug_funcs:
                    npys.append(func(npy))
                #print(len(npys))
                return npys
                


        root_path = 'dataset/'
        mura_train_frame = pd.read_csv(root_path+'MURA-v1.1/train_image_paths.csv')
        mura_val_frame = pd.read_csv(root_path+'MURA-v1.1/valid_image_paths.csv')



        batch_size=25
        data_aug_funcs=[]
        transformer=None
        dim=(224, 224)
        n_channels=3

        k_shot=n_shot

        mura_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part=None, 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        mura_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part=None, 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        humerus_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_HUMERUS', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        humerus_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_HUMERUS', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        forearm_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_FOREARM', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        forearm_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_FOREARM', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        wrist_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_WRIST', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        wrist_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_WRIST', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        shoulder_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_SHOULDER', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        shoulder_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_SHOULDER', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        hand_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_HAND', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        hand_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_HAND', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        finger_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_FINGER', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        finger_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_FINGER', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)

        elbow_train_dataset = MURADataset(mura_frame=mura_train_frame, 
                         root_path=root_path,
                         body_part='XR_ELBOW', 
                         batch_size=batch_size//(len(data_aug_funcs)+1), 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=True, 
                         data_aug_funcs=data_aug_funcs, 
                         normalize=True, 
                         tscl=False,
                         data_distribution=None,
                                        transform=transformer,
                                        k_shot=k_shot)
        elbow_val_dataset = MURADataset(mura_frame=mura_val_frame, 
                         root_path=root_path,
                         body_part='XR_ELBOW', 
                         batch_size=batch_size, 
                         dim=dim, 
                         n_channels=3, 
                         n_classes=2, 
                         shuffle=False, 
                         data_aug_funcs=None, 
                         normalize=True, 
                         data_distribution=None, transform=transformer)



        mura_train_dataloader = DataLoader(mura_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        mura_val_dataloader = DataLoader(mura_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        humerus_train_dataloader = DataLoader(humerus_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        humerus_val_dataloader = DataLoader(humerus_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        forearm_train_dataloader = DataLoader(forearm_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        forearm_val_dataloader = DataLoader(forearm_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        wrist_train_dataloader = DataLoader(wrist_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        wrist_val_dataloader = DataLoader(wrist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        shoulder_train_dataloader = DataLoader(shoulder_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        shoulder_val_dataloader = DataLoader(shoulder_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        hand_train_dataloader = DataLoader(hand_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        hand_val_dataloader = DataLoader(hand_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        finger_train_dataloader = DataLoader(finger_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        finger_val_dataloader = DataLoader(finger_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        elbow_train_dataloader = DataLoader(elbow_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        elbow_val_dataloader = DataLoader(elbow_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)





        train_loader_dict = {'elbow': elbow_train_dataloader, 'finger': finger_train_dataloader,
                       'forearm': forearm_train_dataloader,'hand': hand_train_dataloader,
                      'humerus': humerus_train_dataloader,'shoulder': shoulder_train_dataloader, 'wrist': wrist_train_dataloader}

        val_loader_dict = {'elbow': elbow_val_dataloader, 'finger': finger_val_dataloader,
                       'forearm': forearm_val_dataloader,'hand': hand_val_dataloader,
                      'humerus': humerus_val_dataloader,'shoulder': shoulder_val_dataloader, 'wrist': wrist_val_dataloader}
        

        
        baseline_save_path = 'mura_checkpoints/'+body_part+'_best_densenet169_baseline_seed'+str(seed)+'.pth'
        cutout_save_path = 'mura_checkpoints/'+body_part+'_best_densenet169_cutout_seed'+str(seed)+'.pth'
        save_path = 'mura_checkpoints/'+body_part+'best_densenet169_pg_unet_aug_seed'+str(seed)+'.pth'
        policy_save_path = 'mura_checkpoints/'+body_part+'_best_pg_unet_seed'+str(seed)+'.pth'
        end2end_save_path = 'mura_checkpoints/'+body_part+'_best_end2end_seed'+str(seed)+'.pth'
        gradcam_save_path = 'mura_checkpoints/'+body_part+'_best_gradcam_seed'+str(seed)+'.pth'
      
        train_dataloader = train_loader_dict[body_part]
        val_dataloader = val_loader_dict[body_part]

        baseline_acc, pg_acc, end2end_acc, gradcam_acc = 0, 0, 0, 0, 0
        
        baseline_acc = train(model, 100, optimizer, loss_f, train_dataloader, val_dataloader, baseline_save_path, 
          aug=False, patience=15)

        
        if train_cutout:
            model.load_state_dict(torch.load(baseline_save_path), strict=False)
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            cutout_acc = train(model, 100, optimizer, loss_f, train_dataloader, val_dataloader, cutout_save_path, 
              aug=True, patience=15)

        if train_apga:
            model.load_state_dict(torch.load(baseline_save_path), strict=False)
            optimizer = optim.Adam(model.parameters(), lr=0.00001)

            pg_learning_rate = 0.00001
            gamma = 0.90
            policy = Policy()
            policy = policy.to(device=device)
            pg_optimizer = optim.Adam(policy.parameters(), lr=pg_learning_rate)

            pg_acc = train_with_PG(policy, model, 100, optimizer, pg_optimizer, 
                      loss_f, train_dataloader, val_dataloader, 
                      save_path, policy_save_path, best_val_acc=0, cutout=False, patience=15)

        if train_end2end:
            model.load_state_dict(torch.load(baseline_save_path), strict=False)
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            policy = Policy()
            policy = policy.to(device=device)
            pg_optimizer = optim.Adam(policy.parameters(), lr=pg_learning_rate)
            end2end_acc = train_end2end(policy, model, 100, optimizer, pg_optimizer, 
                  loss_f, elbow_train_dataloader, elbow_val_dataloader, 
                  end2end_save_path, policy_save_path, best_val_acc=0, cutout=False, patience=15, aug_reg=True)
        
        if train_gradcam:
            model.load_state_dict(torch.load(baseline_save_path), strict=False)
            loss_f = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            gradcam_model.load_state_dict(torch.load(baseline_save_path), strict=False)
            gradcam_acc = train_gradcam(model, gradcam_model, 100, optimizer, loss_f, train_dataloader, val_dataloader, gradcam_save_path, 
                gradcam=True, 
                aug=False, patience=15)


        baseline_accs.append(baseline_acc)
        cutout_accs.append(cutout_acc)
        pg_accs.append(pg_acc)
        end2end_accs.append(end2end_acc)
        gradcam_accs.append(gradcam_acc)

        acc_dict = {'baseline_accs': baseline_accs, 'cutout_accs': cutout_accs, 'pg_accs': pg_accs, 
        'end2end_accs': end2end_accs, 'gradcam_accs': gradcam_accs}

        acc_df = pd.DataFrame(acc_dict)
        acc_df.to_csv('mura_checkpoints/'+body_part+'_results_'+str(seed)+'_gradcam.csv')


    


    def show_debug_mask_mura(idx, dataset, body_part='elbow', save=True):
        debug_img, label = dataset.__getitem__(idx)
        debug_mask, mask = predict_masks_from_imgs(policy, 
                                                ToTensor()(np.moveaxis(debug_img, 0, -1)), 
                                                morphology_cleaning=False, largest_connected=False, fill_enclosed=False) 
        if save:
            mask = np.moveaxis(mask, -1, 0)
            show_and_save(debug_mask.numpy(), 
                      'policy_gradient_pngs_mura/'+'masked_'+body_part+'_'+str(i)+'_class_'+str(label)+'.png')
            show_and_save((mask>0.5), 
                      'policy_gradient_pngs_mura/'+'_'+body_part+'_mask'+str(i)+'_class_'+str(label)+'.png', 
                          cmap=plt.get_cmap('gray'))
        else:
            plt.figure()
            plt.imshow(np.moveaxis(debug_mask.numpy(), 0, -1))
            plt.show()
        return label



    def show_debug_img_mura(idx, dataset, body_part='elbow', save=True):
        debug_img, label = dataset.__getitem__(idx)
        if save:
            show_and_save(debug_img, 
                      'policy_gradient_pngs_mura/'+body_part+'_'+str(i)+'_class_'+str(label)+'.png')
        else:
            plt.figure()
            plt.imshow(np.moveaxis(debug_img, 0, -1))
            plt.show()
        return label

    


if __name__ == '__main__': fire.Fire(mura_experiment)
