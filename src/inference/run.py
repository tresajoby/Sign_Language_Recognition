import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.realtime import RealtimeRecognizer


def main():
    print("Starting ASL Recognition System...")
    print("Controls: 'q' = quit | 'r' = reset buffer | 's' = screenshot")
    recognizer = RealtimeRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
