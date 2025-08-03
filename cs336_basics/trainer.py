import logging
import torch
from configs import End2EndConfig
from model import TransformerDecoder
from optimizer import AdamW, CosineScheduler
from utils import load_checkpoint, save_checkpoint, set_seed
from tokenizer import BPETokenizer


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


class Pipeline:
    def __init__(self, config: End2EndConfig = End2EndConfig()):
        logging.info("Initialize Training Pipeline")
        set_seed(config.seed)
        self.config = config
        self.step: int = 0
        self.model = TransformerDecoder(**config.model.to_dict())
        self.optim = AdamW(self.model.parameters(), **config.optim.to_dict())
        self.sched = CosineScheduler(self.optim, **config.sched.to_dict())
        self.token = BPETokenizer.from_vocab(**config.tokens.to_dict())

        self.prepare_model()

    def prepare_model(self):
        logging.info("Prepare Model.")
        logging.info("Move model to %s", self.config.device)
        self.model = self.model.to(self.config.device)
        logging.info("Tie token embedding and LM head weights.")
        self.model.tie_weights()
        self.load_state()

    def train(self,train_data_path:str):
        pass

    @torch.inference_mode()
    def valid(self,val_data_path:str)->float:
        pass

    def save_state(self):
        logging.info("Save the state at step %s to %s", self.step, self.config.trainer.checkpoint_path)
        save_checkpoint(self.model, self.optim, self.step, self.config.trainer.checkpoint_path)

    def load_state(self):
        if self.config.trainer.checkpoint_path.exists():
            logging.info("Load state from %s", self.config.trainer.checkpoint_path)
            self.step = load_checkpoint(self.config.trainer.checkpoint_path, self.model, self.optim)
        else:
            logging.warning("There is no such file: %s", self.config.trainer.checkpoint_path)

    @torch.no_grad()
    def check_forward(self):
        logging.info("Check Forward")
        x = torch.randint(
            0,
            self.config.model.vocab_size,
            (self.config.trainer.batch_size, self.config.model.context_length),
            device=self.config.device,
        )
        return self.model(x)


if __name__ == "__main__":
    pipe = Pipeline()
    # print(pipe.check_forward())
    # pipe.save_state()
    # pipe.load_state()
    # print(pipe.model.token_embeddings)
    # print(pipe.model.lm_head)
    print(pipe.model.lm_head.weight.device)
