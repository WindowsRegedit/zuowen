import os
import sys


def main():
    file_path = os.path.join(os.path.dirname(__file__), "show_ui.py")
    os.system(f"{sys.executable} -m streamlit run {file_path}")


if __name__ == "__main__":
    main()
