# Coin Counter
Coin Counter using Hough Transforms and Neural Networks. Counts total coin value of coins from a live video feed.

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
    $ git clone https://github.com/cyildirim23/EE_4780.git
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

## Use

1. Gather Data for Training and Testing in /train_data/ and /test_data/ folders:

    ```bash
    (.venv) $ poetry run gather
    ```
2. Create Neural Network Model and save in /models/ folder:

    ```bash
    (.venv) $ poetry run create
    ```

3. Train Network until a target is reached (Press Crtl + C to Exit):

    ```bash
    (.venv) $ poetry run train
    ```

4. Use Network to predict worth of coins:

    ```bash
    (.venv) $ poetry run counter
    ```