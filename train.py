import time
from options.train_options import TrainOptions
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    print('opt -> ', opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    loss_side_all = []
    loss_fused_all = []
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        loss_side = []
        loss_fused = []

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size


            model.set_input(data)
            model.optimize_parameters(epoch)

            loss_side.append(model.loss_side.cpu().detach().numpy())
            loss_fused.append(model.loss_fused.cpu().detach().numpy())

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        if epoch == opt.niter + opt.niter_decay:
            print("saving the model of the latest epoch")
            model.save_networks('latest')
        loss_side_all.append(np.sum(np.array(loss_side))/len(loss_side))
        loss_fused_all.append(np.sum(np.array(loss_fused))/len(loss_fused))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    np.save("plot_loss//loss_side_all.npy", np.array(loss_side_all))
    np.save("plot_loss//loss_fused_all.npy", np.array(loss_fused_all))
