# segmentation_prototype
 Repository for unet 3D prototype for parapharyngeal fat pads segmentation


To run first install dependencies:

`pip install -r "requirements.txt"`

Next, run the command:

`python cli.py`


If you want to build an excutable first install pyinstaller:

`pip install pyinstaller`

Next, run the command:

`pyinstaller -c cli.py`

After that copy the /config folder to /dist/cli folder. On that folder you will see a cli.exe file to run the app.