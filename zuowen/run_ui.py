import os
import sys


def main():
    file_path = os.path.join(os.path.dirname(__file__), "ui", "欢_欢迎.py")
    os.system(f"{sys.executable} -m streamlit run {file_path}")


if __name__ == "__main__":
    main()
