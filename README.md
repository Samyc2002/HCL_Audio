# HCL_Audio

To run the code, please install the dependencies first:

```bash
pip install -r requirements.txt
```

Then, for training, run the main.py file:

```bash
python main.py -d dcase19 -n 200 -b 32 --debug True
```

For evaluation, run the evqaluation.py file:

```bash
python evaluation.py -d dcase19 -n 200 -b 32 --debug True
```

For help, run the following:

```bash
python main.py --help
```
