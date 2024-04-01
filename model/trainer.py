from model.model import msstgcn, EarlyStopper
from model.metrics import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve


class Trainer:
    def __init__(self, dil, num_f_maps, dim, num_classes, connections, pool):

        self.model = msstgcn(dil, num_f_maps, dim, num_classes, connections, pool)
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
        self.ce = nn.BCELoss(reduction='mean')


    def train(self, batch_gen, num_epochs, batch_size, learning_rate, device, threshold, weight_decay, num_bin, num_f_maps, pool):
        self.pool = pool
        self.num_f_maps = num_f_maps
        self.model.train()
        self.num_bin = num_bin
        self.model.to(device)
        self.step = [20, 40]
        self.base_lr = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        columns =['Epoch','T F1','T PRAUC','T Loss','T Acc.','T Acc. per Video','T Sns.','T Spc.','T Rcl.','T Prc.','V F1','V PRAUC','V Loss','V Acc.','V Acc. per Video','V Sns.','V Spc.','V Rcl.','V Prc.', 'W.F1']
        report = pd.DataFrame(columns=columns)
        optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        early_stopper = EarlyStopper(patience=5, min_delta=10)

        hyperparamscolumns = ['lr','wd','thr','ftm','nbins', 'pool']


        train_losses = []
        train_accs = []
        train_accs_per_video = []

        Ltrain_f1_scores = []
        Lval_f1_scores = []
        Ltrain_sns_scores = []
        Lval_sns_scores = []
        Ltrain_spcs_scores = []
        Lval_spcs_scores = []
        Ltrain_recs_scores = []
        Lval_recs_scores = []
        Ltrain_prcs_scores = []
        Lval_prcs_scores = []
        Ltrain_PRAUC_scores = []
        Lval_PRAUC_scores = []
        Ltrain_AUCROC_scores = []
        Lval_AUCROC_scores = []

        val_losses = []
        val_accs = []
        val_accs_per_video = []

        sns_predictions_train = np.array([])
        sns_labels_train = np.array([])
        sns_predictions_val = np.array([])
        sns_labels_val = np.array([])
        all_predictions_train = np.array([])
        all_labels_train = np.array([])
        all_predictions_val = np.array([])
        all_labels_val = np.array([])

        curr_epochs = 0
        for epoch in range(num_epochs):

            curr_epoch = epoch

            epoch_loss = 0
            correct = 0
            per_video_correct = 0

            total = 0
            per_video_total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask, weight = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, weight = batch_input.to(device), batch_target.to(device), mask.to(device), weight.to(device)

                per_video_label = torch.unique(batch_target, dim=1).squeeze()
                per_video_label = per_video_label.to(device)

                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes)[:, 1], batch_target.view(-1).float())
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])
                    
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                _, predicted = torch.max(predictions[-1].data, 1)

                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                per_video_predicted = (torch.sum(predicted, dim=1))
                per_video_predicted = per_video_predicted.to(device)
                per_video_predicted_threshold = (per_video_predicted / predicted.shape[1] >= self.threshold).int()
                per_video_predicted_threshold = per_video_predicted_threshold.to(device)

                sns_predictions_train = np.hstack([sns_predictions_train, per_video_predicted_threshold.cpu().numpy().flatten()])
                sns_labels_train = np.hstack([sns_labels_train,per_video_label.cpu().numpy().flatten()])

                all_labels_train = np.hstack([all_labels_train,batch_target.cpu().numpy().flatten()])
                all_predictions_train = np.hstack([all_predictions_train,predictions[:,:, -1, :].detach().cpu().numpy().flatten()])

                per_video_correct += ((per_video_label == per_video_predicted_threshold).float()).sum().item()
                per_video_total += predicted.shape[0]

            train_losses.append(epoch_loss / len(batch_gen.train))
            train_accs.append(float(correct) / total)
            train_accs_per_video.append(float(per_video_correct) / per_video_total)

            ####################################################################################################

            self.model.eval()
            val_loss = 0
            val_correct = 0
            per_video_correct_val = 0
            val_total = 0
            per_video_total_val = 0

            with torch.no_grad():
              val_batch_input, val_batch_target, val_mask, val_weight = batch_gen.next_batch_test(batch_size)
              val_batch_input, val_batch_target, val_mask, val_weight = val_batch_input.to(device), val_batch_target.to(device), val_mask.to(device), val_weight.to(device)
              per_video_label_val = torch.unique(val_batch_target, dim=1).squeeze()
              per_video_label_val = per_video_label_val.to(device)
              val_predictions = self.model(val_batch_input, val_mask)

              for p in val_predictions:
                val_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes)[:, 1], val_batch_target.view(-1).float())
                val_loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

              _, val_predicted = torch.max(val_predictions[-1].data, 1)
              val_correct += ((val_predicted == val_batch_target).float() * val_mask[:, 0, :].squeeze(1)).sum().item()
              val_total += torch.sum(val_mask[:, 0, :]).item()
              per_video_predicted_val = (torch.sum(val_predicted, dim=1))
              per_video_predicted_val = per_video_predicted_val.to(device)
              per_video_predicted_threshold_val = (torch.sum(val_predicted, dim=1) / val_predicted.shape[1] >= self.threshold).int()
              per_video_predicted_threshold_val = per_video_predicted_threshold_val.to(device)
              per_video_correct_val += ((per_video_label_val == per_video_predicted_threshold_val).float()).sum().item()
              per_video_total_val += val_predicted.shape[0]

              sns_predictions_val = np.hstack([per_video_predicted_threshold_val.cpu().numpy().flatten()])
              sns_labels_val = np.hstack([per_video_label_val.cpu().numpy().flatten()])

              all_labels_val = np.hstack([all_labels_val,val_batch_target.cpu().numpy().flatten()])
              all_predictions_val = np.hstack([all_predictions_val,val_predictions[:,:, -1, :].detach().cpu().numpy().flatten()])

              val_losses.append(val_loss.item() / len(batch_gen.test))
              val_accs.append(float(val_correct) / val_total)
              val_accs_per_video.append(float(per_video_correct_val) / per_video_total_val)

            self.model.train()
            curr_epochs += 1
            batch_gen.reset()

            final_train_loss = epoch_loss / len(batch_gen.train)
            final_train_acc = float(correct) / total
            final_train_acc_per_video = float(per_video_correct) / per_video_total
            train_sns, train_spc = sensitivity_specificity(sns_labels_train, sns_predictions_train)
            train_rec, train_prc = recall_precision(sns_labels_train, sns_predictions_train)
            final_val_loss = val_loss / len(batch_gen.test)
            final_val_acc = float(val_correct) / val_total
            final_val_acc_per_video = float(per_video_correct_val) / per_video_total_val
            val_sns, val_spc,  = sensitivity_specificity(sns_labels_val, sns_predictions_val)
            val_rec, val_prc = recall_precision(sns_labels_val, sns_predictions_val)
            train_f1_score = 2 * (train_prc * train_rec) / (train_prc + train_rec) if (train_prc+train_rec > 0) else 0
            val_f1_score = 2 * (val_prc * val_rec) / (val_prc + val_rec) if (val_prc+val_rec > 0) else 0

            t_p,t_r,t_t = precision_recall_curve(all_labels_train, all_predictions_train)
            # Compute PR-AUC
            Ltrain_PRAUC = auc(t_r, t_p)
            v_p,v_r,v_t = precision_recall_curve(all_labels_val, all_predictions_val)
            # Compute PR-AUC
            Lval_PRAUC = auc(v_r, v_p)

            t_roc1, t_roc2, _ = roc_curve(all_labels_train, all_predictions_train)
            Ltrain_AUCROC = auc(t_roc1, t_roc2)
            v_roc1, v_roc2, _ = roc_curve(all_labels_val, all_predictions_val)
            Lval_AUCROC = auc(v_roc1, v_roc2)

            (Ltrain_f1_scores.append(train_f1_score))
            (Lval_f1_scores.append(val_f1_score))
            (Ltrain_sns_scores.append(train_sns))
            (Lval_sns_scores.append(val_sns))
            (Ltrain_spcs_scores.append(train_spc))
            (Lval_spcs_scores.append(val_spc))
            (Ltrain_recs_scores.append(train_rec))
            (Lval_recs_scores.append(val_rec))
            (Ltrain_prcs_scores.append(train_prc))
            (Lval_prcs_scores.append(val_prc))
            (Ltrain_PRAUC_scores.append(Ltrain_PRAUC))
            (Lval_PRAUC_scores.append(Lval_PRAUC))
            (Ltrain_AUCROC_scores.append(Ltrain_AUCROC))
            (Lval_AUCROC_scores.append(Lval_AUCROC))


            #if epoch + 1 == num_epochs:
            #    torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch+1) + ".model")
            #    torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch+1) + ".opt")
            if early_stopper.early_stop(val_loss):
                print(f"Training early stopped at epoch {epoch}.")
                break

        #    if epoch % 10 == 0:
        #          report = pd.concat([report, pd.DataFrame({'Epoch': epoch,
        #          'T F1': train_f1_score,
        #          'T PRAUC': Ltrain_PRAUC,
        #          'T Loss': final_train_loss,
        #          'T Acc.': final_train_acc,
        #          'T Acc. per Video': final_train_acc_per_video,
        #          'T Sns.': train_sns,
        #          'T Spc.': train_spc,
        #          'T Rcl.': train_rec,
        #          'T Prc.': train_prc,
        #          'V F1': val_f1_score,
        #          'V PRAUC': Lval_PRAUC,
        #          'V Loss': final_val_loss.item(),
        #          'V Acc.': final_val_acc,
        #          'V Acc. per Video': final_val_acc_per_video,
        #          'V Sns.': val_sns,
        #          'V Spc.': val_spc,
        #          'V Rcl.': val_rec,
        #          'V Prc.': val_prc
        #      }, index=[0])])
#
        #report.set_index(report.columns[0], inplace=True)
        #display(report)
            
        max_index = np.argmax(Lval_f1_scores)
        final_report = pd.DataFrame(columns=['T Acc. PV', 'T SNS', 'T SPC', 'T AUCROC', 'T PRC', 'T F1', 'T PRAUC', 'V Acc. PV', 'V SNS', 'V SPC', 'V AUCROC', 'V PRC', 'V F1', 'V PRAUC', 'W.F1']+hyperparamscolumns, index=['Avg. (SD)'],
                                    data={
            'T Acc. PV':         np.mean(train_accs_per_video[max_index]),
            'T SNS':             np.mean(Ltrain_sns_scores[max_index]),
            'T SPC':             np.mean(Ltrain_spcs_scores[max_index]),
            'T AUCROC':          np.mean(Ltrain_AUCROC_scores[max_index]),
            'T PRC':             np.mean(Ltrain_prcs_scores[max_index]),
            'T F1':              np.mean(Ltrain_f1_scores[max_index]),
            'T PRAUC':           np.mean(Ltrain_PRAUC_scores[max_index]),
            'V Acc. PV':         np.mean(val_accs_per_video[max_index]),
            'V SNS':             np.mean(Lval_sns_scores[max_index]),
            'V SPC':             np.mean(Lval_spcs_scores[max_index]),
            'V AUCROC':          np.mean(Lval_AUCROC_scores[max_index]),
            'V PRC':             np.mean(Lval_prcs_scores[max_index]),
            'V F1':              np.mean(Lval_f1_scores[max_index]),
            'V PRAUC':           np.mean(Lval_PRAUC_scores[max_index]),
            'W.F1':              np.mean(0.8*Lval_f1_scores[max_index]) - np.mean(0.2*Ltrain_f1_scores[max_index]),
            'lr': self.base_lr,
            'wd': self.weight_decay,
            'thr': self.threshold,
            'ftm': self.num_f_maps,
            'nbins': self.num_bin,
            'pool': self.pool
        })

        print(final_report)

        if True:
            # Create subplots
            fig, axs = plt.subplots(1,4, figsize=(15, 5))
            # Plot accuracy curves
            axs[0].plot(range(1, curr_epochs+1), train_accs, label='Train Acc')
            axs[0].plot(range(1, curr_epochs+1), val_accs, label='Val Acc')
            axs[0].set_title('Train/Val Accuracy')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Accuracy')
            axs[0].legend()

            # Plot loss curves
            axs[1].plot(range(1, curr_epochs+1), train_losses, label='Train Loss')
            axs[1].plot(range(1, curr_epochs+1), val_losses, label='Val Loss')
            axs[1].set_title('Train/Val Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Loss')
            axs[1].legend()

            # Plot precision-recall curve
            axs[2].plot(t_r, t_p, lw=2, label='Train PR curve (area = %0.2f)' % Ltrain_PRAUC)
            axs[2].plot(v_r, v_p, lw=2, label='Validation PR curve (area = %0.2f)' % Lval_PRAUC)
            #axs[2].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            axs[2].set_xlabel('Recall')
            axs[2].set_ylabel('Precision')
            axs[2].set_title('Precision-Recall Curve')
            axs[2].legend()


            # Plot ROC curve
            axs[3].plot(t_roc1, t_roc2, lw=2, label='Train ROC curve (area = %0.2f)' % Ltrain_AUCROC)
            axs[3].plot(v_roc1, v_roc2, lw=2, label='Validation ROC curve (area = %0.2f)' % Lval_AUCROC)
            axs[3].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            axs[3].set_xlim([0.0, 1.0])
            axs[3].set_ylim([0.0, 1.05])
            axs[3].set_xlabel('False Positive Rate')
            axs[3].set_ylabel('True Positive Rate')
            axs[3].set_title('Receiver Operating Characteristic (ROC) Curve')
            axs[3].legend(loc="lower right")


            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show()
        return final_report