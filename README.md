
# ACER for Rogue

  This is a fork of [openai/baselines](https://github.com/openai/baselines)
  intended to use the ACER algorithm on Rogue with the Rogueinabox library.
  In the future we plan to have OpenAI's baselines as dependencies rather
  than have this repo being a fork.

## Cloning and building

  Clone the repository with the default git command:
  ```console
  git clone <URL>
  ```
  
  Before executing the next step, you may want to create/activate your python
  [virtual environment](https://docs.python.org/3/library/venv.html).
  In order to create it:
  ```console
  python3 -m venv /path/to/venv
  ```

  And to activate it:
  ```console
  . /path/to/venv/bin/activate
  ```
  Make sure to do this before the next step, because it will update pip
  and install python dependencies.
  
  Then execute:
  ```console
  make install
  ```
  
  This will install python dependencies and pull and build the Rogueinabox library.

## Manual bulding

  The rogueinabox library is included as a submodule.
  In order for it to be correctly initialized and used, please clone this repo with the following command:
  ```console
  git clone --recurse-submodules <URL>
  ```

  If you cloned without any flags, please run the following command from within your local repo directory:
  ```console
  git submodule update --init --recursive
  ```

  You also need to install python requirements:
  ```console
  pip install -r requirements.txt
  ```

  and build Rogue:
  ```console
  cd rogueinabox_lib
  make
  ```
  
## Training on Rogue

  In order to launch a training on rogue simply execute:
  ```console
  . train_acer.sh -f <cfg_file>
  ``` 
  By default, if you don't provide `-f`, the file `cfg_rogue_default.cfg` will be used.
  Feel free to create your own based on it or on the other configuration files we provide.
  
  For more information:
  ```console
  . train_acer.sh -h
  ``` 
  
## Record a video of your agent playing

  We provide `video_acer.sh` that will load a specified checkpoint and play Rogue on
  a GUI that you can see in real time.
  You can optionally have the script save each frame in plain text format.
  
  For more information:
  ```console
  . video_acer.sh -h
  ```

## Modifications wrt   [openai/baselines](https://github.com/openai/baselines)

  We deleted everything that was not ACER related, i.e.:
  - baselines/acktr
  - baselines/ddpg
  - baselines/deepq
  - baselines/gail
  - baselines/her
  - baselines/ppo1
  - baselines/trpo_mpi
  - data/
  
  We modified:
  - baselines/acer/*
  - baselines/common/cmd_util.py
  - baselines/common/tf_decay.py
  - baselines/common/tf_util.py
  - baselines/common/vec_env/subproc_vec_env.py
  - README.md
  
  And added:
  - baselines/acer/models/*
  - baselines/acer/run_rogue.py
  - baselines/common/vec_env/threaded_vec_env.py
  - envs/
  - openai/
  - rogueinabox_lib/
  - cfg_*
  - AUTHORS
  - LICENSE
  - Makefile
  - requirements.txt
  - train_acer.sh
  - video_acer.sh
