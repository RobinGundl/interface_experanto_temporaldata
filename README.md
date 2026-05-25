# interface_experanto_temporaldata
interface_experanto_temporaldata is a Python package that enables data exchange between Experanto and Temporaldata.

## Installation
```bash
git clone https://github.com/RobinGundl/interface_experanto_temporaldata
cd interface_experanto_temporaldata
pip install -e .
```

## Usage
```python
# Create an Interface instance from a dictionary
example_dict = Interface({
    "example_1": experanto_data,
    "example_2": temporaldata_data,
})

# Convert data from Experanto format to Temporaldata format
example_conversion_1 = example_dict.process_as_temporaldata(
    "example_1"
)

# Convert data from Temporaldata format to Experanto format
example_conversion_2 = example_dict.process_as_experanto(
    "example_2"
)
```

## Data
This package uses data from the Sensorium Competition (see: https://github.com/sensorium-competition) and brainsets (see: https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.PerichMillerPopulation2018.html)

