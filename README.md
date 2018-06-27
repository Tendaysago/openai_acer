
# ACER for Rogue

  This is a fork of [openai/baselines](https://github.com/openai/baselines)
  intended to use the ACER algorithm on Rogue with the Rogueinabox library.
  In the future we plan to have OpenAI's baselines as dependencies rather
  than this repo being a fork.

## Cloning and building

  Clone the repository with the default git command:
  ```console
  git clone <URL>
  ```
  
  Then execute:
  ```console
  make install
  ```
  
  This will pull and build the Rogueinabox library.

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
  cd roguelib_module
  make
  ```
