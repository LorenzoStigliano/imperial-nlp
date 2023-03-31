import os
import re
from typing import Any, Optional, Tuple

from nltk.stem import PorterStemmer

from .config import MODEL_FOLDER, MODEL_NAME, BASELINE_PATH, BASELINE_DF_NAME, Array_like
from .data_processing import DataProcessing

import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, roc_curve, roc_auc_score, recall_score


def weighted_binary_cross_entropy(output: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]=None) -> torch.Tensor:

    if weight is not None:
        assert len(weight) == 2
        loss = weight[1] * (target * torch.log(output)) + \
               weight[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


class LJABERT(nn.Module):
    """
    Our custom model, LJABERT, where LJA stands for Lorenzo, Jorge, Adrien
    """

    def __init__(self, checkpoint: str = "roberta-base", num_labels: int = 1, weight: Optional[torch.Tensor] = None):
        super(LJABERT, self).__init__()
        self.num_labels = num_labels

        # Load Roberta
        self.model = AutoModel.from_pretrained(
            checkpoint,
            config=AutoConfig.from_pretrained(
                checkpoint, output_attentions=False, output_hidden_states=False
            ),
        )
        # Freeze Roberta's weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Other layers
        self.classifier = nn.Linear(768, num_labels)
        # self.dropout = nn.Dropout(0.1)
        self.activation_function = nn.Sigmoid()

        # Loss function for the classification task
        self.loss_function = nn.BCELoss(weight=weight)  # binary classification

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Feed Roberta
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Extract the last hidden states
        out = outputs.last_hidden_state

        # For this classification problem, we are only interested in the hidden
        # state associated with the initial token [CLS]
        # This state is known to capture the semantics of the entire sentence
        # better than the other states, hence the [0 ... ] below
        out = self.classifier(out[:, 0, :].view(-1, 768))

        # Sigmoid as an activation layer for final classification prediction
        out = self.activation_function(out)
        logits = out.view(-1, self.num_labels).float()

        # Calculate the loss
        loss = None
        if labels is not None:
            loss = weighted_binary_cross_entropy(logits, labels.view(-1, 1).float())

        # Note to self: Add this to output if we calculate them and want to store them
        # , hidden_states=outputs.hidden_states, attentions=outputs.attentions
        return TokenClassifierOutput(loss=loss, logits=logits)
    
class BagOfWordsClassifier():
    def __init__(self):
        self.counts = {}
        self.counts_PCL = {}
        self.counts_not_PCL = {}
        self.PCL_word_count = 0
        self.no_PCL_word_count = 0
        self.PCL_document_count = 0
        self.no_PCL_document_count = 0

    def clean_text_tokenize(self, text):
        stop_words = ['this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'don', "don't", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'shan', "shan't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

        text.lower()
        # removing " at start of sentences
        text = text.strip("\"")
        # replacing repetitions of punctations
        text = re.sub(r'\"+', '', text)

        # Tokenize links
        text = re.sub(r'https? : \S+', '[WEBSITE]', text)
        # removing referencing on usernames with @
        text = re.sub(r'@\S+', '', text)
        # removing smileys with : (like :),:D,:( etc)
        text = re.sub(r':\S+', '', text)
        # Remove punctation
        text = re.sub(r"[!.,;:?\'\"\´]", "", text)
        text = re.sub('(?<![\w])20[0-5][0-9]-?[0-9]*',
                    '[YEAR]', text)              # Year token
        text = re.sub('(?<![\w])1[0-9]{3}-?[0-9]*',
                    '[YEAR]', text)                 # Year token
        # replacing numbers with [NUM] tag  eg 1,000, 1.32, 5-7. Assert these numbers are not inside words (i.e. H1, )
        text = re.sub('(?<![\w])[0-9]+[.,]?[0-9]*(?![\w])', '[NUM]', text)
        text = re.sub('\[NUM\]-\[NUM\]', '[NUM]', text)
        # Again to delete account numbers lol 12-5223-231
        text = re.sub('\[NUM\]-\[NUM\]', '[NUM]', text)
        text = re.sub('(?<=\[NUM\])-(?=[a-zA-Z])', ' ', text)
        text = re.sub('[ ]*', ' ', text)
        text = re.sub('<h>', '.', text)

        porter = PorterStemmer()
        words = text.split()
        for i, word in enumerate(words):
            if word in stop_words:
                words.pop(i)
            else:
                words[i] = porter.stem(word)
        return words

    def train(self, train_DF):
        for i, row in train_DF.iterrows():
            text = row["text"]
            label = row["binary_label"]

            if label == 0:
                self.no_PCL_document_count += 1
            else:
                self.PCL_document_count += 1

            words = self.clean_text_tokenize(text)
            for word in words:
                self.counts[word] = 1 + \
                    (self.counts[word] if word in self.counts.keys() else 0)

                if label == 0:
                    self.no_PCL_word_count += 1
                    self.counts_not_PCL[word] = 1 + \
                        (self.counts_not_PCL[word]
                         if word in self.counts_not_PCL.keys() else 0)
                else:
                    self.PCL_word_count += 1
                    self.counts_PCL[word] = 1 + \
                        (self.counts_PCL[word]
                         if word in self.counts_PCL.keys() else 0)
                    
    def predict(self, sentences):

        prior = self.PCL_document_count / \
            (self.PCL_document_count + self.no_PCL_document_count)
        epsilon = 1  # epsilon smoothing
        if type(sentences) is str:
            sentences = [sentences]
        if type(sentences) is pd.DataFrame:
            i, sentences = sentences.iterrows()

        predictions = []
        for sentence in sentences:

            likelihood = 1
            for word in sentence:
                class_count = self.counts_PCL[word] if word in self.counts_PCL.keys(
                ) else 0
                likelihood *= (class_count+epsilon) / \
                    (len(self.counts) + self.PCL_word_count)

            prob_PCL = prior*likelihood

            likelihood = 1
            for word in sentence:
                class_count = self.counts_not_PCL[word] if word in self.counts_not_PCL.keys(
                ) else 0
                likelihood *= (class_count+epsilon) / \
                    (len(self.counts) + self.no_PCL_word_count)

            prob_not_PCL = (1-prior)*likelihood

            predictions.append(1 if prob_PCL > prob_not_PCL else 0)

        return predictions



class NLPModel(DataProcessing):
    def __init__(self) -> None:
        super().__init__()
        # Cuda
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        print(f"Device: {self.device}")

        # Class weights
        df_cw = self.preprocessed_data_pcl_train  # to evaluate class weights
        self.class_weights = torch.tensor(
            [1 - (len(df_cw[df_cw["binary_label"] == 0]) / len(df_cw)),
             1 - (len(df_cw[df_cw["binary_label"] == 1]) / len(df_cw))]
        ).to(self.device)
        # Model
        self.model_folder = MODEL_FOLDER
        self.create_folder(self.model_folder)
        self.model_name = MODEL_NAME
        self.model_path = os.path.join(self.model_folder, self.model_name)
        self.model = self.load_model(self.model_path)
        # Baseline
        self.baseline_path = BASELINE_PATH
        self.baseline_df_name = BASELINE_DF_NAME
        self.baseline_df_path = os.path.join(
            self.data_folder, self.baseline_df_name
        )
        self.baseline_df = self.load_data(self.baseline_df_path)
        # self.baseline_model = self.load_baseline(self.baseline_path)
        # Model training parameters
        self.training_metrics = None
        self.verbose_train = True
        self.print_val_every = 1
        self.batch_size = 64
        self.num_epochs = 5
        self.init_lr = 1e-4

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def load_model(self, path: str) -> Optional[Any]:
        if os.path.exists(path):
            print("Loading existing LJABERT model ...")
            model = LJABERT(weight=self.class_weights)
            model.load_state_dict(torch.load(path))
            model.eval()
            return model
        print("No existing LJABERT model was found.")
        return None

    def save_model(self, model: Any, path: str) -> None:
        torch.save(model.state_dict(), path)

    # helper function to save predictions to an output file
    def labels2file(self, p: Array_like, outf_path: str):
        with open(outf_path, "w") as outf:
            for pi in p:
                outf.write(",".join([str(k) for k in pi]) + "\n")

    def load_baseline(self, path: str) -> Any:
        if os.path.exists(path):
            print("Loading baseline ...")
            model = ClassificationModel("roberta", path, use_cuda=self.cuda_available)
            return model
        # otherwise, train the baseline
        return self.create_baseline()

    def create_baseline(self):
        print("Creating baseline ...")
        baseline_model_args = ClassificationArgs(
            num_train_epochs=1, no_save=False, no_cache=True, overwrite_output_dir=True
        )
        baseline_model = ClassificationModel(
            "roberta",
            "roberta-base",
            args=baseline_model_args,
            num_labels=2,
            use_cuda=self.cuda_available,
        )
        # not processed dataset, train set
        # df = self.data_pcl_train[["text", "binary_label"]]
        # train the baseline on the exact same dataframe
        # that has been shuffle and downsampled
        df = self.baseline_df[["text", "label"]]
        eval_df = self.data_pcl_dev[["text", "binary_label"]]
        baseline_model.train_model(
            df, eval_df=eval_df, f1_score=f1_score, acc=accuracy_score
        )
        self.baseline_model = baseline_model
        # no need to save, it is saved automatically to the right folder
        # by Classification Model
        return baseline_model

    def calculate_f1_score(
        model: Any, X_test: Array_like, y_test: Array_like
    ) -> float:
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, pos_label=1)

    def calculate_roc(
        model: Any, X_test: Array_like, y_test: Array_like
    ) -> float:
        y_pred = model.predict(X_test)

        roc_score = roc_auc_score(y_test, y_pred)
        print(f"The roc score is {roc_score}")
        return roc_curve(y_test, y_pred, pos_label=1), roc_score

    def create_train_val_split(self) -> Tuple[Array_like, Array_like]:
        # TODO to complete
        train_set, val_set = ()
        return train_set, val_set

    """
    def hyperparameter_tuning(self):
        train_set, val_set = self.create_train_test_split()
        def objective(trial):
            # save memory and prevent from training an already trained model
            try:
                del model
                del optimizer
            except Exception as _:
                ()
            scheduler_name = trial.suggest_categorical("scheduler_name", ["custom", "exponential"])
            batch_size_trial = trial.suggest_int("batch_size_trial", 16, 128)
            init_lr = trial.suggest_float(
                "init_lr", 1e-5, 0.1, log=True,
            )
            loader_train = DataLoader(train_set, batch_size=batch_size_trial, shuffle=True, num_workers=2)
            loader_val = DataLoader(val_set, batch_size=batch_size_trial, shuffle=True, num_workers=2)
            model = LJA()
            nb_epochs = 20
            weight_decay = 1e-7
            # Adam optimizer & learning rate scheduler
            scheduler, optimizer = get_pair_scheduler_optimizer(model, scheduler_name, init_lr, nb_epochs=nb_epochs, init_lr=init_lr, weight_decay=weight_decay)
            train_part(model, optimizer, loader_train, epochs=nb_epochs, scheduler=scheduler)
            torch.save(model.state_dict(), f"models/model_{trial.number}.pt")
            return check_accuracy(loader_val, model, analysis=False, verbose=False)
    """

    def create_train_val_dataloader(self, df: pd.DataFrame, eval_df: pd.DataFrame):
        df_train = df.copy(deep=True)
        df_train["merge"] = (
            df_train["input_ids"]
            + df_train["attention_mask"]
            + df_train["binary_label"].apply(lambda x: [x])
        )
        df_val = eval_df.copy(deep=True)
        df_val["merge"] = (
            df_val["input_ids"]
            + df_val["attention_mask"]
            + df_val["binary_label"].apply(lambda x: [x])
        )
        dataset_train = [torch.tensor(x).long() for x in df_train["merge"].tolist()]
        dataset_val = [torch.tensor(x).long() for x in df_val["merge"].tolist()]

        train_dataloader = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        eval_dataloader = DataLoader(
            dataset_val, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        return train_dataloader, eval_dataloader

    def run_model(self) -> Any:
        if self.model is not None:
            return self.model
        # ELSE, FIT OUR MODEL!
        print("Training LJABERT model...")

        # Extract the train/dev split, preprocessed, and use the input ids and attention masks!
        df = self.preprocessed_data_pcl_train[
            ["input_ids", "attention_mask", "binary_label"]
        ]
        eval_df = self.preprocessed_data_pcl_dev[
            ["input_ids", "attention_mask", "binary_label"]
        ]

        model = LJABERT(weight=self.class_weights).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.init_lr, eps=1e-8)
        lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

        # Train the model
        train_dataloader, eval_dataloader = self.create_train_val_dataloader(
            df, eval_df
        )

        # Training metrics
        training_metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_acc": [],
            "val_acc": [],
            "train_bal_acc": [],
            "val_bal_acc": [],
            "train_rocauc": [],
            "val_rocauc": [],
            "train_prec": [],
            "val_prec": [],
            "train_recall": [],
            "val_recall": [],
        }
        training_metrics_epoch = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_acc": [],
            "val_acc": [],
            "train_bal_acc": [],
            "val_bal_acc": [],
            "train_rocauc": [],
            "val_rocauc": [],
            "train_prec": [],
            "val_prec": [],
            "train_recall": [],
            "val_recall": [],
        }
        for epoch in tqdm(list(range(self.num_epochs))):
            train_loss_epoch = 0
            val_loss_epoch = 0
            train_f1_epoch = 0
            val_f1_epoch = 0
            train_acc_epoch = 0
            val_acc_epoch = 0
            train_bal_acc_epoch = 0
            val_bal_acc_epoch = 0
            train_rocauc_epoch = 0
            val_rocauc_epoch = 0
            train_prec_epoch = 0
            val_prec_epoch = 0
            train_recall_epoch = 0
            val_recall_epoch = 0

            model.train()
            for data in tqdm(train_dataloader):
                data = data.to(self.device)
                input_ids, attention_mask, labels = (
                    data[:, : self.MAX_SEQ_LEN],
                    data[:, self.MAX_SEQ_LEN : 2 * self.MAX_SEQ_LEN],
                    data[:, 2 * self.MAX_SEQ_LEN :],
                )
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs.loss
                training_metrics["train_loss"].append(loss.item())
                train_loss_epoch += loss.item()
                loss.backward()

                logits = outputs.logits
                y_pred = torch.round(logits).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                # metrics
                train_f1 = f1_score(labels, y_pred)
                train_acc = accuracy_score(labels, y_pred)
                train_bal_acc = balanced_accuracy_score(labels, y_pred)
                train_prec = precision_score(labels, y_pred)
                train_rocauc = roc_auc_score(labels, y_pred)
                train_recall = recall_score(labels, y_pred)
                training_metrics["train_f1"].append(train_f1)
                training_metrics["train_acc"].append(train_acc)
                training_metrics["train_bal_acc"].append(train_bal_acc)
                training_metrics["train_prec"].append(train_prec)
                training_metrics["train_rocauc"].append(train_rocauc)
                training_metrics["train_recall"].append(train_recall)
                train_f1_epoch += train_f1
                train_acc_epoch += train_acc
                train_bal_acc_epoch += train_bal_acc
                train_rocauc_epoch += train_rocauc
                train_prec_epoch += train_prec
                train_recall_epoch += train_recall

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss_epoch /= len(train_dataloader)
            train_f1_epoch /= len(train_dataloader)
            train_acc_epoch /= len(train_dataloader)
            train_bal_acc_epoch /= len(train_dataloader)
            train_rocauc_epoch /= len(train_dataloader)
            train_prec_epoch /= len(train_dataloader)
            train_recall_epoch /= len(train_dataloader)
            training_metrics_epoch["train_loss"].append(train_loss_epoch)
            training_metrics_epoch["train_f1"].append(train_f1_epoch)
            training_metrics_epoch["train_acc"].append(train_acc_epoch)
            training_metrics_epoch["train_bal_acc"].append(train_bal_acc_epoch)
            training_metrics_epoch["train_prec"].append(train_prec_epoch)
            training_metrics_epoch["train_rocauc"].append(train_rocauc_epoch)
            training_metrics_epoch["train_recall"].append(train_recall_epoch)

            model.eval()
            for data in tqdm(eval_dataloader):
                data = data.to(self.device)
                input_ids, attention_mask, labels = (
                    data[:, : self.MAX_SEQ_LEN],
                    data[:, self.MAX_SEQ_LEN : 2 * self.MAX_SEQ_LEN],
                    data[:, 2 * self.MAX_SEQ_LEN :],
                )
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, labels)

                loss = outputs.loss
                training_metrics["val_loss"].append(loss.item())
                val_loss_epoch += loss.item()

                logits = outputs.logits
                y_pred = torch.round(logits).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                # metrics
                val_f1 = f1_score(labels, y_pred)
                val_acc = accuracy_score(labels, y_pred)
                val_bal_acc = balanced_accuracy_score(labels, y_pred)
                val_prec = precision_score(labels, y_pred)
                val_rocauc = roc_auc_score(labels, y_pred)
                val_recall = recall_score(labels, y_pred)
                training_metrics["val_f1"].append(val_f1)
                training_metrics["val_acc"].append(val_acc)
                training_metrics["val_bal_acc"].append(val_bal_acc)
                training_metrics["val_prec"].append(val_prec)
                training_metrics["val_rocauc"].append(val_rocauc)
                training_metrics["val_recall"].append(val_recall)
                val_f1_epoch += val_f1
                val_acc_epoch += val_acc
                val_bal_acc_epoch += val_bal_acc
                val_rocauc_epoch += val_rocauc
                val_prec_epoch += val_prec
                val_recall_epoch += val_recall

            val_loss_epoch /= len(eval_dataloader)
            val_f1_epoch /= len(eval_dataloader)
            val_acc_epoch /= len(eval_dataloader)
            val_bal_acc_epoch /= len(eval_dataloader)
            val_rocauc_epoch /= len(eval_dataloader)
            val_prec_epoch /= len(eval_dataloader)
            val_recall_epoch /= len(eval_dataloader)
            training_metrics_epoch["val_loss"].append(val_loss_epoch)
            training_metrics_epoch["val_f1"].append(val_f1_epoch)
            training_metrics_epoch["val_acc"].append(val_acc_epoch)
            training_metrics_epoch["val_bal_acc"].append(val_bal_acc_epoch)
            training_metrics_epoch["val_prec"].append(val_prec_epoch)
            training_metrics_epoch["val_rocauc"].append(val_rocauc_epoch)
            training_metrics_epoch["val_recall"].append(val_recall_epoch)

            if self.verbose_train:
                if epoch % self.print_val_every == 0:
                    print(
                        f"Epoch {epoch}\ntrain loss: {train_loss_epoch} val loss: {val_loss_epoch}\ntrain f1: {train_f1_epoch} val f1: {val_f1_epoch}\ntrain acc: {train_acc_epoch} val acc: {val_acc_epoch}\ntrain bal_acc: {train_bal_acc_epoch} val bal_acc: {val_bal_acc_epoch}\ntrain prec: {train_prec_epoch} val prec: {val_prec_epoch}\ntrain rocauc: {train_rocauc_epoch} val rocauc: {val_rocauc_epoch}\ntrain recall: {train_recall_epoch} val recall: {val_recall_epoch}"
                    )
                else:
                    print(f"Epoch {epoch}\ntrain loss: {train_loss_epoch} train f1: {train_f1_epoch} train acc: {train_acc_epoch} train bal_acc: {train_bal_acc_epoch} train prec: {train_prec_epoch} train rocauc: {train_rocauc_epoch} train recall: {train_recall_epoch}")

        # store the training metrics
        self.training_metrics = pd.DataFrame.from_dict(training_metrics)
        self.training_metrics_epoch = pd.DataFrame.from_dict(training_metrics_epoch)

        # save the metrics
        self.training_metrics.to_csv(os.path.join(self.data_folder, "training_metrics.csv"), index=False)
        self.training_metrics_epoch.to_csv(os.path.join(self.data_folder, "training_metrics_epoch.csv"), index=False)

        # Save the model
        self.model = model
        self.save_model(model, self.model_path)
        return model
