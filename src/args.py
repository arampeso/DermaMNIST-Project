import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "custom"])

    parser.add_argument("-output_dir", type=str, default="sessions")

    parser.add_argument("-split_dir", type=str, default="data/splits")

    parser.add_argument("-epochs", type=int, default=20)

    parser.add_argument("-batch_size", type=int, default=32,
                        choices=[16, 32, 64, 128])

    parser.add_argument("-lr", type=float, default=1e-3,
                        choices=[1e-4, 1e-3])

    parser.add_argument("-wd", type=float, default=1e-5,
                        choices=[1e-5, 1e-7])

    args = parser.parse_args()
    return args