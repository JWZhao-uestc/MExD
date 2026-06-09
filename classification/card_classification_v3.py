import logging
import time
import gc
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel
from tqdm import tqdm
from data_loader import *
from ema import EMA
#from model import *
from model_v1 import *
from pretraining.encoder import Model as AuxCls
from pretraining.resnet import ResNet18
from utils import *
from diffusion_utils import *
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve

plt.style.use('ggplot')

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
    #for c in range(0, 1):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        # print(label, prediction,label.shape, prediction.shape, labels.shape, predictions.shape)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        #print("threshold", threshold)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        if config.diffusion.apply_aux_cls:
            if config.data.dataset == "gaussian_mixture":
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(1, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1)
                ).to(self.device)
            # elif config.data.dataset == "MNIST" and config.model.arch == "simple":
            elif config.data.dataset == "MNIST":
                self.cond_pred_model = nn.Sequential(
                    nn.Linear(config.model.data_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.ReLU(),
                    nn.Linear(300, 100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(),
                    nn.Linear(100, config.data.num_classes)
                ).to(self.device)
            
            
            # zjw revise
            elif config.data.dataset.split('-')[-1] == 'WSI':
                
                if 'Camelyon16' in config.data.dataset:
                    from MOE_v3_2 import MixtralSparseMoeBlock   ##### replace your model #C16
                if 'TCGA' in config.data.dataset:
                    from MOE_v3_3 import MixtralSparseMoeBlock   ##### replace your model #tcga
                if 'BRACS' in config.data.dataset:
                    from MOE_v3_4 import MixtralSparseMoeBlock   ##### replace your model #bracs
                
                milnet = MixtralSparseMoeBlock(config) #.cuda()
            
                self.cond_pred_model = milnet.to(self.device)


            elif config.data.dataset in ["FashionMNIST", "CIFAR10", "CIFAR100"]:
                if config.diffusion.aux_cls.arch == "lenet":
                    self.cond_pred_model = LeNet(config.data.num_classes,
                                                 config.model.n_input_channels,
                                                 config.model.n_input_padding).to(self.device)
                elif config.diffusion.aux_cls.arch == "lenet5":
                    self.cond_pred_model = LeNet5(config.data.num_classes,
                                                  config.model.n_input_channels,
                                                  config.model.n_input_padding).to(self.device)
                elif config.diffusion.aux_cls.arch == "resnet18_ckpt":
                    # self.cond_pred_model = resnet18(pretrained=False).to(self.device)
                    self.cond_pred_model = ResNet18(num_classes=config.data.num_classes).to(self.device)
                else:
                    self.cond_pred_model = AuxCls(config).to(self.device)
            else:
                self.cond_pred_model = AuxCls(config).to(self.device)
            self.aux_cost_function = nn.CrossEntropyLoss()
        else:
            pass

        # scaling temperature for NLL and ECE computation
        self.tuned_scale_T = None


    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred = self.compute_guiding_prediction(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1, #config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1, #config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.CrossEntropyLoss()
        brier_score = nn.MSELoss()

        # apply an auxiliary optimizer for the guidance classifier
        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):  # load saved auxiliary classifier
                # aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                #                                      config.diffusion.trained_aux_cls_ckpt_name),
                #                         map_location=self.device)
                aux_states = torch.load(config.diffusion.trained_aux_cls_ckpt_path, map_location=self.device)
                print('You have load feature exactor correctly!', config.diffusion.trained_aux_cls_ckpt_path)
                mag = self.cond_pred_model.load_state_dict(aux_states, strict=True)
                print('missing_keys', mag.missing_keys)
                self.cond_pred_model.eval()
            

            elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                self.cond_pred_model.eval()
            else:  # pre-train the guidance auxiliary classifier
                assert config.diffusion.aux_cls.pre_train
                self.cond_pred_model.train()
                pretrain_start_time = time.time()
                for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                    for feature_label_set in train_loader:
                        if config.data.dataset == "gaussian_mixture":
                            x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                        else:
                            y_labels_batch, x_batch = feature_label_set
                            y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch,
                                                                                                  config)
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                            y_one_hot_batch.to(self.device),
                                                                            aux_optimizer)
                    if epoch % config.diffusion.aux_cls.logging_interval == 0:
                        logging.info(
                            f"epoch: {epoch}, guidance auxiliary classifier pre-training loss: {aux_loss}"
                        )
                pretrain_end_time = time.time()
                logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                    (pretrain_end_time - pretrain_start_time) / 60))
                # save auxiliary model after pre-training
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            
        #exit()
        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt_best.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0], strict=True) #

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = 0.0
            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, feature_label_set in enumerate(train_loader):
                    
                    # read origin feature
                    y_labels_batch, x_batch = feature_label_set
                    y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)

                    #print('x_batch', x_batch.shape, 'y_labels_batch', y_labels_batch.shape)

                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                    n = x_batch.size(0)
                    
                    # if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                    #     x_batch = torch.flatten(x_batch, 1)
                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss
                    # record unflattened x as input to guidance aux classifier
                    x_unflat_batch = x_batch #.view(-1, config.data.feat_size).to(self.device)
                    
                    output, _ = self.cond_pred_model(x_unflat_batch.to(self.device))
            
                    #y_0_hat_batch, bag_feature ,attention = output['Y_prob'], output["Bag_feature"], output["A"]
                    y_0_hat_batch, expert_feature = output['e_Y_prob'], output['expert_feature']
                    expert_feature = expert_feature.unsqueeze(0)
                    
                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                   
                    
                    # conditional diffusion model zjw commented

                    output = model(expert_feature.to(self.device), y_t_batch, t, y_0_hat_batch)
                
                    loss = (e - output).square().mean()  # use the same noise sample e during training to compute loss

                    # cross-entropy for y_0 reparameterization
                    loss0 = torch.tensor([0])
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                        loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                    f"Noise Estimation loss: {loss.item()}, " +
                                    f"data time: {data_time / (i + 1)}"
                            )
                        )

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # joint train aux classifier along with diffusion model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(x_unflat_batch.to(self.device), y_one_hot_batch,
                                                                            aux_optimizer)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, guidance auxiliary classifier joint-training loss: {aux_loss}"
                            )

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                            if step > 1:  # skip saving the initial ckpt
                                torch.save(
                                    aux_states,
                                    os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                )
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    data_start = time.time()
                    #break

                logging.info(
                    (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, " +
                     f"data time: {data_time / (i + 1)}")
                )

                # Evaluate
                if epoch % self.config.training.validation_freq == 0 \
                        or epoch + 1 == self.config.training.n_epochs:
                    if config.data.dataset == "toy":
                        with torch.no_grad():
                            model.eval()
                            label_vec = nn.functional.one_hot(test_dataset[:][1]).float().to(self.device)
                            # prior mean at timestep T
                            test_y_0_hat = self.compute_guiding_prediction(
                                test_dataset[:][0].to(self.device)).softmax(dim=1)
                            y_T_mean = test_y_0_hat
                            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                y_T_mean = torch.zeros(test_y_0_hat.shape).to(test_y_0_hat.device)
                            if epoch == start_epoch:
                                fig, axs = plt.subplots(1, self.num_figs,
                                                        figsize=(self.num_figs * 8.5, 8.5), clear=True)
                                for i in range(self.num_figs - 1):
                                    cur_y = q_sample(label_vec.cpu(), y_T_mean.cpu(),
                                                     self.alphas_bar_sqrt.cpu(),
                                                     self.one_minus_alphas_bar_sqrt.cpu(),
                                                     torch.tensor([i * self.vis_step])).detach().cpu()
                                    axs[i].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                    axs[i].set_title('$q(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=25)
                                cur_y = q_sample(label_vec.cpu(), y_T_mean.cpu(),
                                                 self.alphas_bar_sqrt.cpu(),
                                                 self.one_minus_alphas_bar_sqrt.cpu(),
                                                 torch.tensor([self.num_timesteps - 1])).detach().cpu()
                                axs[self.num_figs - 1].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                axs[self.num_figs - 1].set_title(
                                    '$q(\mathbf{y}_{' + str(self.num_timesteps - 1) + '})$', fontsize=25)
                                if not tb_logger is None:
                                    tb_logger.add_figure('data', fig, step)
                            y_seq = p_sample_loop(model, test_dataset[:][0].to(self.device),
                                                  test_y_0_hat, y_T_mean,
                                                  self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt,
                                                  only_last_sample=False)
                            fig, axs = plt.subplots(1, self.num_figs,
                                                    figsize=(self.num_figs * 8.5, 8.5), clear=True)
                            cur_y = y_seq[0].detach().cpu()
                            axs[self.num_figs - 1].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                            axs[self.num_figs - 1].set_title('$p({y}_\mathbf{prior})$', fontsize=25)
                            for i in range(self.num_figs - 1):
                                cur_y = y_seq[self.num_timesteps - i * self.vis_step - 1].detach().cpu()
                                axs[i].scatter(cur_y[:, 0], cur_y[:, 1], s=10, c=test_dataset[:][1]);
                                axs[i].set_title('$p(\mathbf{x}_{' + str(self.vis_step * i) + '})$', fontsize=25)
                            acc_avg = accuracy(y_seq[-1].detach().cpu(), test_dataset[:][1].cpu())[0]
                            logging.info(
                                (f"epoch: {epoch}, step: {step}, Average accuracy: {acc_avg}%")
                            )
                            if not tb_logger is None:
                                tb_logger.add_figure('samples', fig, step)
                                tb_logger.add_scalar('accuracy', acc_avg.item(), global_step=step)
                            fig.savefig(
                                os.path.join(args.im_path, 'samples_T{}_{}.pdf'.format(self.num_timesteps, step)))
                            plt.close()
                    else:
                        model.eval()
                        self.cond_pred_model.eval()
                        test_labels = []
                        test_preds = []
                        test_prob = []
                        acc_avg = 0. 
                    
                        for test_batch_idx, (target, images) in enumerate(test_loader):
                            images_unflat = images.to(self.device)
                            # if config.data.dataset == "toy" \
                            #         or config.model.arch == "simple" \
                            #         or config.model.arch == "linear":
                            #     images = torch.flatten(images, 1)
                            images = images.view(-1, config.data.feat_size).to(self.device)
                            target = target.to(self.device)
                            test_labels.extend(target.cpu().numpy())
                            
                            with torch.no_grad():
                                
                                images = images[None,:,:]
                                output, _ = self.cond_pred_model(images.to(self.device))
            
                                target_pred = output['e_Y_prob']
                                expert_feature = output['expert_feature']
                                
                                
                                # prior mean at timestep T
                                y_T_mean = target_pred
                                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                                
                            
                                label_t_0 = p_sample_loop(model, expert_feature, target_pred, y_T_mean,
                                                          self.num_timesteps, self.alphas,
                                                          self.one_minus_alphas_bar_sqrt,
                                                          only_last_sample=True)
                                
                                bag_prediction = torch.softmax(label_t_0, -1)
                                test_prob.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
                                bag_pred = torch.argmax(label_t_0, dim=1)
                                test_preds.extend(bag_pred.cpu().numpy())


                        test_labels = np.array(test_labels)
        
                        test_predictions = np.array(test_preds)

                        labels = []
                        for i in test_labels:
                            label = np.zeros(config.data.num_classes)
                            label[i] = 1
                            labels.append(label)
                        labels = np.array(labels)
                        test_prob = np.array(test_prob)
                        auc_value, _, thresholds_optimal = multi_label_roc(labels, test_prob, config.data.num_classes, pos_label=1)
                 
                        auc_value = sum(auc_value)/len(auc_value)*100

                        bag_score = 0
                        # average acc of all labels
                        for i in range(0, len(test_loader)):
                            #print(test_labels[i], test_predictions[i])
                            if np.array_equal(test_labels[i], test_predictions[i]) == 0:
                                print(i)
                            bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
                        acc_avg = (bag_score / len(test_loader)) *100  #ACC
                        
                        cls_report = classification_report(test_labels, test_predictions, digits=4)
                        print('\n', cls_report)
                        if (acc_avg+auc_value)/2 > max_accuracy:
                            logging.info("Update best accuracy at Epoch {}.".format(epoch))
                            torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        max_accuracy = max(max_accuracy, (acc_avg+auc_value)/2)
                        if not tb_logger is None:
                            tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, " +
                                    f"Average accuracy: {acc_avg}, " +
                                    f"Average AUC: {auc_value:.2f}%"
                                    f"Max accuracy: {max_accuracy:.2f}%" 

                            )
                        )

            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

    def test(self):
        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)
        dataset_object, train_dataset, test_dataset = get_dataset(args, config)
        # use test batch size for training set during inference
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=1, #config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,#config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                
                aux_states = torch.load(config.diffusion.trained_aux_cls_ckpt_path, map_location=self.device)
                print('You have load feature exactor correctly!')
                self.cond_pred_model.load_state_dict(aux_states, strict=True)
                #self.cond_pred_model.eval()
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.eval()
        # diff model
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        
        model.load_state_dict(states[0], strict=True)
        states[1]["param_groups"][0]["eps"] = self.config.optim.eps
        
        if args.sanity_check:
            logging.info("Evaluation function implementation sanity check...")
            config.testing.n_samples = 10
        if args.test_sample_seed >= 0:
            logging.info(f"Manually setting seed {args.test_sample_seed} for test time sampling of class prototype...")
            set_random_seed(args.test_sample_seed)
        
        model.eval()
        self.cond_pred_model.eval()
        test_labels = []
        test_preds = []
        test_prob = []
        acc_avg = 0
        with torch.no_grad():
            
            for step, feature_label_set in tqdm(enumerate(test_loader)):
                #x_batch, y_labels_batch = feature_label_set
                y_labels_batch, x_batch = feature_label_set
                label = y_labels_batch.numpy()
                
                output, _ = self.cond_pred_model(x_batch.to(self.device))
                target_pred = output['e_Y_prob']
                expert_feature = output['expert_feature']
                
                target = y_labels_batch.to(self.device)
                
                # prior mean at timestep T
                y_T_mean = target_pred
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                
                label_t_0 = p_sample_loop(model, expert_feature, target_pred, y_T_mean,
                                            self.num_timesteps, self.alphas,
                                            self.one_minus_alphas_bar_sqrt,
                                            only_last_sample=True)
                
                acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()       
                label_t_0 = torch.softmax(label_t_0, -1)
                
                
                bag_prediction = label_t_0
                max_prediction = bag_prediction
                e_bag_pred = torch.argmax(bag_prediction, -1)
                test_labels.extend(label)
                
                test_preds.extend(e_bag_pred.cpu().numpy())
                test_prob.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])

        acc_avg /= (step + 1)
       
        logging.info("Update best accuracy at Epoch {:.2f}.".format(acc_avg))
                            

        test_labels = np.array(test_labels)
        
        test_predictions = np.array(test_preds)
      
        
        labels = []
        for i in test_labels:
            label = np.zeros(config.data.num_classes)
            label[i] = 1
            labels.append(label)
        labels = np.array(labels)
        test_prob = np.array(test_prob)
        auc_value, _, thresholds_optimal = multi_label_roc(labels, test_prob, config.data.num_classes, pos_label=1)
        

        log_path = os.path.join(args.log_path, "test.txt")
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n *****************Threshold by optimal*****************')
        
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels,test_predictions))
        info = confusion_matrix(test_labels,test_predictions)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))

        bag_score = 0
        # average acc of all labels
        for i in range(0, len(test_loader)):
            
            bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
        avg_score = bag_score / len(test_loader)  #ACC
        
        cls_report = classification_report(test_labels, test_predictions, digits=4)

        print('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        print('\n', cls_report)
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
            log_txt.write('\n' + cls_report)
                