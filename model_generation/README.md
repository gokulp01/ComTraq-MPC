# Model generation and evaluation

### Note: This is the legacy version of the model generation and evaluation. The new version is available in the [comtraq-mpc](../src/comtraq-mpc/) folder.

This folder is the cpp variant for generating the model of the ground vehicle. For the python variant, please refer to the [python folder](../src/comtraq-mpc/).

## Dependencies

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [pybind11](https://pybind11.readthedocs.io/en/stable/)

## Build

```bash
mkdir build
cd build
cmake ..
make
```

or simply run the `build_model.sh` script:

```bash
chmod +x build_model.sh
./build_model.sh
```

The `config.cpp` file contains the parameters for the model. The `main.cpp` file contains the main function to generate the model.

You can run an example of the model by running the `model.py` script which has the python interface for the cpp model.
