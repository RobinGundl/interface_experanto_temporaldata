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

## Example Data
`pipeline_example.h5` in `example_pipelines/Temporaldata/data/processed/test` contains transformed sample data derived from the `sensorium-competition/experanto` project, which is licensed under the MIT License.

