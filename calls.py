import keras
import configparser
import os

class SaveCall(keras.callbacks.Callback):
    ini = 'ckpt.ini'
    section = 'default'
    name = 'latest_checkpoint'
    iepoch = 'epoch'
    ibatch = 'batch'
    epoch_mode = 'epoch_mode'
    batch_mode = 'batch_mode'
    train_mode = 'train_mode'

    def __init__(self,filepath,period = 1,mode = epoch_mode,max_one = True):
        '''
        :param filepath: file saved path
        :param period: save period
        :param mode: mode to save
                    epoch_mode: save per period epoch
                    batch_mode,save per period batch,when batch start,restart count
                    train_mode,save per period batch,only initial when train start
        :param max_one: only have one file
        '''
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.period = period
        self.max_one = max_one

    def load(self,model):
        '''
        load model weights

        :param model: Sequentail
        :return: initial epoch
        '''
        ckpt_dir = os.path.dirname(self.filepath)
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        self.ini_path = os.path.join(ckpt_dir, self.ini)
        self.conf = configparser.ConfigParser()

        if os.path.exists(self.ini_path):
            self.conf.read(self.ini_path)
            ckpt_path = self.conf.get(self.section, self.name)
            if os.path.exists(ckpt_path):
                model.load_weights(ckpt_path)
                epoch = self.conf.getint(self.section, self.iepoch)
                print("load weight from file {},start with epoch {}".format(ckpt_path,epoch))
                return epoch
        else:
            self.conf.add_section(self.section)

        return 0

    def save(self,epoch,logs=None):
        if self.mode == self.epoch_mode:
            epoch += 1
        name = self.filepath.format(epoch=epoch, **logs)
        self.model.save_weights(name)
        print("{} has saved".format(name))

        #delete last checkpoint
        if self.max_one:
            if self.conf.has_section(self.section) and self.conf.has_option(self.section,self.name):
                old_ckpt = self.conf.get(self.section,self.name)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

        self.conf.set(self.section, self.name, name)
        self.conf.set(self.section, self.iepoch, str(epoch))
        with open(self.ini_path, 'w') as f:
            self.conf.write(f)

    def on_epoch_end(self, epoch, logs=None):
        if self.mode == self.epoch_mode:
            if self.train_count % self.period == 0:
                self.save(epoch, logs)
            self.train_count += 1

    def on_batch_end(self, batch, logs=None):
        if self.mode == self.batch_mode or self.mode == self.train_mode:
            if self.train_count % self.period == 0:
                self.save(self.epoch_on_batch,logs)
            self.train_count += 1

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_on_batch = epoch
        if self.mode == self.batch_mode:
            self.train_count = 1


    def on_train_begin(self, logs=None):
        if self.mode == self.train_mode or self.mode == self.epoch_mode:
            self.train_count = 1