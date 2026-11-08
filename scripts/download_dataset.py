import argparse
from agentgraph.dataset import download_hotpotqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    path = download_hotpotqa(
        limit=args.limit
    )

    print(f"Dataset saved to: {path}")


if __name__ == "__main__":
    main()
