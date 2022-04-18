import argparse
from ..datasets.NoiseGenerator import NoiseGenerator
from .Experiment import Experiment
from ..datasets.muscima.Muscima import Muscima
from ..datasets.SegmentationDescription import SegmentationDescription
from ..datasets.Datasets import Datasets
from ..datasets.DatasetOptions import DatasetOptions
from ..datasets.SegmentationDescription import SegmentationDescription


class Ex_Datasets(Experiment):
    @property
    def name(self):
        return "datasets"

    def describe(self):
        return """
        Used to validate the dataset pipeline logic
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--dataset_seed", default=0, type=int, help="Dataset-slicing random seed.")
        parser.add_argument("--dataset", default="exploration", type=str, help="Dataset name.")
        parser.add_argument("--val_pages", default=10, type=int, help="Validation page count.")
        parser.add_argument("--sup_pages", default=10, type=int, help="Supervised page count.")
        parser.add_argument("--sup_repeat", default=1, type=int, help="Repeat supervised dataset.")
        parser.add_argument("--unsup_pages", default=50, type=int, help="Unsupervised page count.")
        parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
        parser.add_argument("--noise_size_ss", default=2, type=float, help="Noise size in staff space multiple.")
        parser.add_argument("--noise_dropout", default=0.25, type=float, help="Noise dropout percentage.")
        parser.add_argument("--symbol", default="notehead", type=str, help="Symbol to segment.")

        parser.add_argument("--no_plot", default=False, action="store_true", help="Do not plot the dataset content.")
        
    def run(self, args: argparse.Namespace):
        noise = NoiseGenerator(
            seed=args.dataset_seed,
            max_noise_size=int(Muscima.DPSS * args.noise_size_ss),
            dropout_ratio=args.noise_dropout,
            largest_tiles_only=True
        )
        opts = DatasetOptions(
            dataset_seed=args.dataset_seed,
            tile_size_wh=(512, 256),
            validation_pages=args.val_pages,
            supervised_pages=args.sup_pages,
            unsupervised_pages=args.unsup_pages,
            supervised_repeat=args.sup_repeat,
            batch_size=args.batch_size,
            segdesc=SegmentationDescription.from_name(args.symbol),
            unsupervised_transformation=noise.dataset_transformation,
            verbose=True # print more info during dataset slicing
        )
        print("OPTS:", vars(opts))
        print()
        ds_train, ds_validate, ds_test = Datasets.by_name(args.dataset, opts)

        if args.no_plot:
            return

        print("Plotting...")
        import matplotlib.pyplot as plt
        for (xl, xu), (yl, yu) in ds_train.take(5).as_numpy_iterator():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.suptitle("ds_train dataset content, batch_item 0, channel 0")
            ax1.imshow(xl[0,:,:,0]) ; ax1.set_xlabel("X_labeled")
            ax3.imshow(xu[0,:,:,0]) ; ax3.set_xlabel("X_unlabeled")
            ax2.imshow(yl[0,:,:,0]) ; ax2.set_xlabel("Y_labeled")
            ax4.imshow(yu[0,:,:,0]) ; ax4.set_xlabel("Y_unlabeled")
            plt.show()

        for x, y in ds_validate.take(5).as_numpy_iterator():
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("ds_validate dataset content, batch_item 0, channel 0")
            ax1.imshow(x[0,:,:,0]) ; ax1.set_ylabel("X")
            ax2.imshow(y[0,:,:,0]) ; ax2.set_ylabel("Y")
            plt.show()

        for x, y in ds_test.take(5).as_numpy_iterator():
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("ds_test dataset content, batch_item 0, channel 0")
            ax1.imshow(x[0,:,:,0]) ; ax1.set_ylabel("X")
            ax2.imshow(y[0,:,:,0]) ; ax2.set_ylabel("Y")
            plt.show()
