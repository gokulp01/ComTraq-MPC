# multi-agent

To run the code:

```
pip install -r requirements.txt
```
Please note: you have to install torch separately

Also install: pybind11 to interface C++ with Python using: `pip install pybind11`

Then build the model
```
mkdir build 
cd build 
cmake ..
cmake --build .
```

`model_demo.py` gives a demo file to interact with the model