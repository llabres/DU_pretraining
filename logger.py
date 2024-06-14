"""
modified from original: https://github.com/rubenpt91/MP-DocVQA-Framework/blob/master/logger.py
"""

import os, socket, datetime, getpass
import wandb as wb


class Logger:

    def __init__(self, config):

        self.use_wandb = config['wandb']
        self.log_folder = config['save_dir']

        project_name = config['project_name']

        experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        
        self.experiment_name = config.get('experiment_name', f"{config['model_name']}_{config['dataset_name']}__{experiment_date}")

        machine = socket.gethostname()

        dataset = config['dataset_name']
        page_retrieval = config.get('page_retrieval', '-').capitalize()
        visual_encoder = config.get('visual_module', {}).get('model', '-').upper()

        document_pages = config.get('max_pages', None)
        page_tokens = config.get('page_tokens', None)
        tags = [config['model_name'], dataset, machine]
        config = {'Model': config['model_name'], 'Weights': config['model_weights'], 'Dataset': dataset,
                  'Page retrieval': page_retrieval, 'Visual Encoder': visual_encoder,
                  'Batch size': config['batch_size'], 'Max. Seq. Length': config.get('max_sequence_length', '-'),
                  'lr': config['lr'], 'seed': config['seed'], 'Model Config': config.get('model_config', '-')}

        if document_pages:
            config['Max Pages'] = document_pages

        if page_tokens:
            config['PAGE tokens'] = page_tokens


        if self.use_wandb:
            self.logger = wb.init(project=project_name, name=self.experiment_name, dir=self.log_folder, tags=tags, config=config)
        self._print_config(config)

        self.current_epoch = 0
        self.best_metric = 0
        self.len_dataset = 0

    def _print_config(self, config):
        print(f"{config['Model']}: {config['Weights']} \n{{")
        for k, v in config.items():
            if k != 'Model' and k != 'Weights' and k != 'Model Config':
                print(f"\t{k}: {v}")
        print("}\n")

    def log_model_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.use_wandb:
            self.logger.config.update({
                'Model Params': int(total_params / 1e6),  # In millions
                'Model Trainable Params': int(trainable_params / 1e6)  # In millions
            })

        print(f"Model parameters: {total_params:d} - Trainable: {trainable_params:d} ({trainable_params / total_params * 100:2.2f}%)")

    def update_global_metrics(self, metrics, best_metric_name):
        if metrics[best_metric_name] > self.best_metric:
            self.best_metric = metrics[best_metric_name]
            self.best_epoch = self.current_epoch
            return True

        else:
            return False


    def log_val_metrics(self, metrics, update_best=False, best_metric_name=None):
        assert not update_best or best_metric_name is not None, "Best metric must be provided if update_best is True"

        best_metric_name = f"Val/Epoch {best_metric_name}"
        
        str_msg = f"Epoch {self.current_epoch:d}: " + " - ".join([f"{k}: {v:2.4f}" for k, v in metrics.items()])
        metrics = {f"Val/Epoch {k}": v for k, v in metrics.items()}
        if self.use_wandb:
            self.logger.log(metrics, step=self.current_epoch*self.len_dataset + self.len_dataset)

        if update_best:
            str_msg += f"\tBest {best_metric_name}: {metrics[best_metric_name]:2.4f}"

            if self.use_wandb:
                self.logger.config.update({
                    f"Best {best_metric_name}": metrics[best_metric_name],
                    "Best epoch": self.current_epoch
                }, allow_val_change=True)

        print(str_msg)