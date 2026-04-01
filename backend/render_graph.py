import subprocess
import sys
from pathlib import Path

MODULE = "app.bot.chat.visualize_graph"
MERMAID_FILE = Path("graph.mmd")
PNG_FILE = Path("graph.png")


def generate_mermaid() -> None:
    print("Generating Mermaid diagram...")
    result = subprocess.run(
        [sys.executable, "-m", MODULE, "--mermaid", str(MERMAID_FILE)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Failed to generate Mermaid:")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        raise SystemExit(1)

    if result.stdout:
        print(result.stdout.strip())
    print("Mermaid file created")


def convert_to_png() -> None:
    print("Converting to PNG...")
    cmd = f'npx @mermaid-js/mermaid-cli -i "{MERMAID_FILE}" -o "{PNG_FILE}"'
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print("Failed to convert to PNG:")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        raise SystemExit(1)

    if result.stdout:
        print(result.stdout.strip())
    print(f"PNG created: {PNG_FILE.resolve()}")


def main() -> None:
    print(f"Using Python: {sys.executable}")
    generate_mermaid()
    convert_to_png()


if __name__ == "__main__":
    main()