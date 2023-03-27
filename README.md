# Coin Counter
Coin Counter using Hough Transforms and Neural Networks. Counts Coins from a live video feed.

## Installation

1. Install pyenv:
   * [Windows](https://github.com/pyenv-win/pyenv-win)
   * [Linux](https://github.com/pyenv/pyenv#automatic-installer)
   * [macOS](https://github.com/pyenv/pyenv#homebrew-in-macos)

2. Install Python using pyenv 

    ```bash
    $ pyenv install 3.11.1
    ```

3. Clone the repositories using git. Make sure to check out the develop branch of both repositories to get the most recent changes.

    ```bash
    $ git clone https://github.com/lsu-ece/pecs.git
    $ git clone https://github.com/lsu-ece/mad-macs.git
    ```

3. Create/Activate a Python Virtual Environment while inside the directory.

    ```bash
    $ python -m venv .venv

    # windows
    $ ./venv/Scripts/activate

    # unix
    $ ./venv/bin/activate

    # your prompt should change to:
    (.venv) $
    ```

4. Update pip and install Poetry 

    ```bash
    (.venv) $ python -m pip install -U pip poetry
    (.venv) $ poetry install
    ```
5. Install Tensorflow separately due to known bug:

    ```bash
    (.venv) $ pip install tensorflow
    ```