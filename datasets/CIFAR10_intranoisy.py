from __future__ import print_function
import os
import sys
import errno
import numpy as np
from PIL import Image
import torch.utils.data as data
import contextlib
import pickle


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import torchvision.datasets.accimage as accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def build_set(root, train, imgs, noise_type='pairflip', noise_rate=0.5):
    """
       Function to return the lists of paths with the corresponding labels for the images
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If true, returns the list pertaining to training images and labels, else otherwise
    Returns:
        return_list: list of 236_comb_fromZeroNoise-tuples with 1st location specifying path and 2nd location specifying the class
    """

    tmp_imgs = imgs

    if train:
        with temp_seed(0):
            if noise_type == 'pairflip':
                label_noises = np.random.multinomial(1, [0.55, 0.45], len(imgs)).argmax(1)
            else:
                noise_rate_per_class = noise_rate / (10-1)
                noise_dist = np.ones(10) * noise_rate_per_class
                noise_dist[0] = 1 - noise_rate
                label_noises = np.random.multinomial(1, noise_dist, len(imgs)).argmax(1)
                print(label_noises)

        noise_labeled_imgs = [(x, (y + label_noises[i]) % 10) for i, (x, y) in enumerate(imgs)]
        for i in range(100):
            print('original target is : ', imgs[i][1], 'noised target is : ', noise_labeled_imgs[i][1])
        return noise_labeled_imgs
    else:
        return imgs


class cifar(data.Dataset):
    """`cifar10 <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>`_ Dataset.
    Args:
        root (string): Root directory of dataset the images and corresponding lists exist
            inside raw folder
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = []
    raw_folder = 'raw'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root,  train=True, transform=None, target_transform=None, download=False, loader=default_loader,
                 noise_type='pairflip', noise_rate=0.45):

        #self.root = os.path.expanduser('../' + root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.loader = loader


        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.imgs = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(os.path.join(self.root, self.raw_folder), 'cifar-10-batches-py/', file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.imgs.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC
        self.imgs = list(zip(self.imgs, targets))

        self.load_meta()

        self.imgs = build_set(os.path.join(self.root, self.raw_folder),
                                                               self.train, self.imgs, noise_type, noise_rate)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # path, target = self.imgs[index]
        # img = self.loader(path)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     img = self.target_transform(img)
        #
        # return img, target

        img = self.imgs[index][0]
        #img = self.loader(path)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            img = self.target_transform(img)

        return (img, *self.imgs[index][1:])


    def _check_exists(self):
        pth = os.path.join(self.root, self.raw_folder)
        return os.path.exists(os.path.join(pth, 'cifar-10-batches-py'))

    def __len__(self):
        return len(self.imgs)

    def download(self):
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            tar = tarfile.open(file_path, 'r')
            for item in tar:
                tar.extract(item, file_path.replace(filename, ''))
            os.unlink(file_path)

        print('Done!')

    def load_meta(self):
        path = os.path.join(self.root, self.raw_folder,'cifar-10-batches-py/',self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}.values()


