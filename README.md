# rock-steady

Motion selectivity via stabilization

# Setting up Airsim

[First, install Airsim environments locally](https://github.com/microsoft/AirSim/releases). Currently supported are the 
environments:

* nh
* mountains
* trapcam
* blocks (flaky)

Fire up one environment. This will create a local configuration folder in `~/Documents/AirSim`. Copy either `references/settings-big.json`
or `references/settings-small.json` into `~/Documents/AirSim/settings.json` depending on whether you want to generate high-res
movies (for presentations) or low-res hdf5 files (for ML).

# Setting up the environment

Setting up the environment should be as easy as `pip install -r requirements.txt`. 

# Generating sequences

There is a makefile that you can use to start environments, but it only runs on Ubuntu, and 
docs warn that AirSim works best on Windows. On Windows, start the target environment manually (double-click on the exe).
You may create a shortcut prior to this that with the following flags: `-ResX=640 -ResY=480 -windowed`

Then run `src/data/command_airsim.py` with the appropriate flags to generate sequences. The sequences will be dumped to `data/raw` as hdf5 files.

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
