from lightning.pytorch.callbacks import BaseFinetuning


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        for module in list(pl_module.model.children())[:-1]:
            self.freeze(module)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            for module in list(pl_module.model.children())[:-1]:
                self.unfreeze_and_add_param_group(
                    modules=module,
                    optimizer=optimizer,
                    train_bn=True,
                )
