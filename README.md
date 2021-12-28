# Information Retrieval System

## Initialisation

Creates three files in folder `ir_data` representing a dictionary and a inverted file from the dataset.

- `dictionary.npy`: 2D numpy array containing terms, their position in the inverted file and the number of documents they appear in.
- `document_ids`: Array of longs of the inverted file's document identifiers.
- `frequencies`: Array of bytes of the inverted file's frequencies.

| Option | Description                     |
| ------ | ------------------------------- |
| -h     | Show help.                      |
| -p     | Path to the dataset.            |
| -P     | Print percentage of completion. |

## IR System

The file `ir_system.py` is the main library of the IR System. It can also be used as a standalone program to search a query given a dictionary, document identifiers and term frequencies obtained from an initialisation file.

The implemented IR System is a standard boolean model using tf-idf.

| Option | Description                                       |
| ------ | ------------------------------------------------- |
| -h     | Show help.                                        |
| -d     | Filename of dictionary.                           |
| -i     | Filename of inverted file's document identifiers. |
| -f     | Filename of inverted file's frequencies.          |
| -D     | Name of the dataset from which to use its according dictionary and inverted file. |
| -n     | Number of returned relevant documents.            |

## Test

Test files serve to compare results from the IR System with the dataset's expected results.

For each query, its identifier will be printed out with its index in parenthesis, followed by the number of expected relevant documents returned out of the number of expected relevant documents.

If a word in the query is not part of the dictionary, a warning is printed out: `Word 'example' not in dictionary. Skipping...` This does not affect the program's state.

| Option | Description                                       |
| ------ | ------------------------------------------------- |
| -h     | Show help.                                        |
| -p     | Path to the dataset.                              |
| -n     | Number of returned relevant documents.            |

## Example

This example uses the Cranfield datatset. It is possible to download it here: http://ir.dcs.gla.ac.uk/resources/test_collections/cran/

The downloaded files are then placed under the following folder: `datasets/cranfield/` (not datatsets/cran/).

1. Initialisation

After creating an initialisation program according to the dataset's format, running it will create three files associated to this dataset.

`python ./initialise_cranfield.py -p`

2. IR System

Using the IR System, it is possible to search a query among all documents of the dataset and to retrieve the best suited ones. In this example, the 10 best ones will be returned.

`python ./ir_system.py -d=cranfield_dictionary.npy -i=cranfield_document_ids -f=cranfield_frequencies -n=10`

Instead of giving each file for the dictionary and the inverted file, it is possible to only give the name of the dataset:

`python ./ir_system.py -D=cranfield -n=10`

3. Test

Testing the IR System against the dataset's expected results is done using the next command line after creating the test program.

`python ./test_cranfield.py -p='datasets/cranfield/'`
